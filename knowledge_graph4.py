import chromadb
import ast
import networkx as nx
import os
from pprint import pprint
import google.generativeai as genai
import concurrent.futures

GOOGLE_API_KEY = 'AIzaSyCSadZXV7kV8tcT3oAB119XNHvidf3Ij_M'
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')


class ChromaDBManager:
    def __init__(self, collection_name):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name=collection_name)

    def add_documents(self, documents, metadatas, ids):
        processed_documents = [str(d) for d in documents]
        self.collection.add(documents=processed_documents,
                            metadatas=metadatas, ids=ids)
        pprint(self.query_documents("Sample Query", 1))

    def query_documents(self, query_texts, n_results):
        return self.collection.query(query_texts=query_texts, n_results=n_results)


class CodeAnalyzerPass1(ast.NodeVisitor):
    def __init__(self, graph, source_code):
        self.graph = graph
        self.source_code = source_code
        self.current_scope = []

    def _get_full_name(self, name):
        return '.'.join(self.current_scope + [name])

    def _add_node(self, node, entity_type):
        name = node.name
        full_name = self._get_full_name(name)
        code_segment = ast.get_source_segment(self.source_code, node)
        self.graph.add_node(full_name, type=entity_type, code=code_segment)
        return full_name

    def visit_ClassDef(self, node):
        class_name = self._add_node(node, 'class')
        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_FunctionDef(self, node):
        func_name = self._add_node(node, 'function')
        if self.current_scope:
            parent = '.'.join(self.current_scope)
            self.graph.add_edge(parent, func_name)
        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()


class CodeAnalyzerPass2(ast.NodeVisitor):
    def __init__(self, graph, defined_entities):
        self.graph = graph
        self.current_scope = []
        self.defined_entities = defined_entities
        self.instance_assignments = {}

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            instance_var = self._get_var_name(node.targets[0])
            class_name = node.value.func.id
            self.instance_assignments[instance_var] = class_name
        self.generic_visit(node)

    def _get_var_name(self, node):
        return node.id if isinstance(node, ast.Name) else None

    def _get_full_name(self, name):
        return '.'.join(self.current_scope + [name])

    def visit_ClassDef(self, node):
        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_FunctionDef(self, node):
        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()

    def _resolve_call(self, call_name):
        if call_name == 'self' and len(self.current_scope) > 1:
            return self.current_scope[-2]
        for i in range(len(self.current_scope), -1, -1):
            scoped_name = '.'.join(self.current_scope[:i] + [call_name])
            if scoped_name in self.defined_entities:
                return scoped_name
        return None

    def visit_Call(self, node):
        caller = '.'.join(
            self.current_scope) if self.current_scope else 'global'

        if isinstance(node.func, ast.Name):
            callee = self._resolve_call(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in self.defined_entities:
                    callee = self._resolve_call(
                        f"{node.func.value.id}.{node.func.attr}")
                elif node.func.value.id == 'self':
                    callee = self._resolve_call(
                        f"{self.current_scope[-2]}.{node.func.attr}")
                else:
                    callee = self._resolve_call(
                        f"{node.func.value.id}.{node.func.attr}")
            else:
                callee = None
        elif isinstance(node.func, ast.Lambda):
            callee = 'lambda'
        else:
            callee = None

        if callee and self.graph.has_node(callee):
            self.graph.add_edge(caller, callee)

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            instance_var = node.func.value.id
            method_name = node.func.attr
            class_name = self.instance_assignments.get(instance_var)

            if class_name:
                callee = f"{class_name}.{method_name}"
                if callee in self.defined_entities:
                    caller = '.'.join(
                        self.current_scope) if self.current_scope else 'global'
                    self.graph.add_edge(caller, callee)
            else:
                callee = f"unknown_class.{method_name}"
                if "unknown_class" in callee:
                    self.graph.add_node(callee, type='method')
                    caller = '.'.join(
                        self.current_scope) if self.current_scope else 'global'
                    self.graph.add_edge(caller, callee)

        self.generic_visit(node)


def analyze_code(code):
    tree = ast.parse(code)
    graph = nx.DiGraph()
    defined_entities = set()

    analyzer_pass1 = CodeAnalyzerPass1(graph, code)
    analyzer_pass1.visit(tree)

    for node in graph.nodes:
        defined_entities.add(node)

    analyzer_pass2 = CodeAnalyzerPass2(graph, defined_entities)
    analyzer_pass2.visit(tree)

    return graph


def parse_python_file(file_path):
    with open(file_path, "r") as file:
        code = file.read()
    return code


def build_graph_from_directory(directory_path):
    graph = nx.DiGraph()
    directory_node_name = os.path.basename(directory_path)
    graph.add_node(directory_node_name, type='directory')

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                code = parse_python_file(file_path)
                file_graph = analyze_code(code)

                subdirectory_node_name = os.path.relpath(
                    root, directory_path) if root != directory_path else None

                if subdirectory_node_name:
                    graph.add_node(subdirectory_node_name, type='subdirectory')
                    graph.add_edge(directory_node_name, subdirectory_node_name)

                    file_node_name = os.path.join(subdirectory_node_name, file)
                else:
                    file_node_name = file

                graph.add_node(file_node_name, type='file')
                graph.add_edge(subdirectory_node_name, file_node_name) if subdirectory_node_name else graph.add_edge(
                    directory_node_name, file_node_name)

                for node, data in file_graph.nodes(data=True):
                    new_node_name = f"{directory_node_name}/{subdirectory_node_name}/{file_node_name}/{node}" if subdirectory_node_name else f"{directory_node_name}/{file_node_name}/{node}"

                    if 'code' in data and 'unknown_class' not in new_node_name:
                        graph.add_node(new_node_name, **data)

                for edge in file_graph.edges():
                    new_edge = tuple(
                        f"{directory_node_name}/{subdirectory_node_name}/{file_node_name}/{n}" if subdirectory_node_name else f"{directory_node_name}/{file_node_name}/{n}" for n in edge)
                    graph.add_edge(*new_edge)

    return graph


def remove_unknown_class_nodes(graph):
    nodes_to_remove = [node for node in graph.nodes if 'unknown_class' in node]
    graph.remove_nodes_from(nodes_to_remove)
    return graph


def get_node_info_string(node, graph):
    node_data = graph.nodes[node]
    node_type = node_data.get('type', 'unknown')
    code = node_data.get('code', None)
    linked_functions = [neighbor for neighbor in graph.successors(
        node) if graph.nodes[neighbor].get('type') == 'function']
    predecessors = list(graph.predecessors(node))
    children = [child for child in graph.successors(node)]

    info_string = f"Node: {node}, Type: {node_type}\n"
    info_string += f"Code:\n{code if code is not None else 'None'}\n"
    info_string += f"Linked Functions:\n{linked_functions if linked_functions else 'None'}\n"
    info_string += f"Predecessors:\n{predecessors if predecessors else 'None'}\n"
    info_string += f"Children:\n{children if children else 'None'}\n"

    return info_string


def generate_docstrings_batch(nodes, graph):
    prompts = []
    info_strings = []

    for node in nodes:
        node_data = graph.nodes[node]
        code = node_data.get('code', None)
        info_string = get_node_info_string(node, graph)
        
        info_strings.append({"node": node, "info_string": info_string})

    # Convert info_strings to a string representation
    info_strings_str = ",\n".join([f'{{"node": {info["node"]}, "docstring": """{info["info_string"]}"""}}' for info in info_strings])

    prompt = """
    Create comprehensive docstrings with the following fields: 
    - **Description**: Briefly explain the purpose and functionality of the directory, file, class, or function.

    - **Parameters**: List and describe the input parameters, if applicable, including their types and any constraints.

    - **Return Value(s)**: Specify the return values, if applicable, along with their types and meanings.

    - **Examples**: Provide usage examples that illustrate how to use the directory, file, class, or function.

    - **Notes**: Include any additional information, caveats, or special considerations.

    Here is a sample docstring:

    ```docstring
    Set the IP (Instant Processing) adapter for the Stable Diffusion XL InstantID pipeline.

    This function configures the IP adapter for cross-attention processors in the UNet model.

    Parameters:
    - model_ckpt (str): Path to the checkpoint file for loading the IP adapter's state_dict.
    - num_tokens (int): Number of tokens used in cross-attention processing.
    - scale (float): Scaling factor for the IP attention processor.

    Return Value(s):
    - None

    Examples:

        ```
        pipeline = StableDiffusionXLInstantIDPipeline()
        pipeline.set_ip_adapter("model_checkpoint.pth", num_tokens=256, scale=0.5)
        ```

        Notes:
        - This function sets the IP adapter based on the provided checkpoint file, number of tokens,
        and scale factor.
        - It utilizes the UNet model's attention processors for cross-attention configurations.
        - The state_dict of the IP adapter is loaded from the provided checkpoint file.
        ```

    Provide the generated docstrings in JSON array format as the output. Required Output Format:

    [
        {"node": 1, "docstring": "Generated docstring for node 1..."},
        {"node": 2, "docstring": "Generated docstring for node 2..."},
        ...
    ]

    IMPORTANT NOTES: 
    - The docstring should not contain a direct copy of the array of the Nodes. Instead, generate content explaining what the Node does
    - The docstring should not contain any code, especially a copy of the code of a function.
    - The docstring should follow the advice set above, laying out the Description, Parameters, Return Types, Examples and Notes. 

    Here is your array of Nodes: 
    """

    prompts.append(prompt)
    prompts.append(info_strings_str)
    responses = model.generate_content(prompts)


    for node, response in zip(nodes, responses):
        print(f"DOCSTRING for Node {node}: {response.text}")


class CodeFlowManager:
    def __init__(self, repo_path, collection_name):
        self.repo_path = repo_path
        self.collection_name = collection_name
        self.graph = None

    def run_code_flow_analysis(self):
        self.graph = build_graph_from_directory(self.repo_path)
        remove_unknown_class_nodes(self.graph)

    def _generate_docstrings_batch(self, nodes, graph):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(
                generate_docstring, node, graph): node for node in nodes}
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                try:
                    docstring = future.result()
                    print(f"Generated docstring for node: {node}")
                    # Update your graph or perform any other necessary actions with the docstring
                except Exception as e:
                    print(f"Error generating docstring for node {node}: {e}")

    def add_code_info_to_chroma_db(self, batch_size=4):
        documents = []
        metadatas = []
        ids = []

        all_nodes = list(self.graph.nodes)

        for i in range(0, len(all_nodes), batch_size):
            batch_nodes = all_nodes[i:i + batch_size]
            generate_docstrings_batch(batch_nodes, self.graph)


# Example usage
repo_path = '/Users/dhruvroongta/PycharmProjects/basics/EmptyFolder'
collection_name = 'your_collection_name'
code_flow_manager = CodeFlowManager(repo_path, collection_name)
code_flow_manager.run_code_flow_analysis()
code_flow_manager.add_code_info_to_chroma_db()
