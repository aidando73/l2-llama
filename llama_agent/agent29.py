import os
from typing import Literal, Optional, Tuple, Union
import re
import json
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.shared_params.sampling_params import SamplingParams
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.api.datatypes import StopReason
from llama_models.llama3.api.tool_utils import (
    is_valid_python_list,
    parse_python_list_for_function_calls,
)
from llama_models.llama3.api.datatypes import (
    ToolDefinition,
    ToolParamDefinition,
)
from llama_models.llama3.prompt_templates.system_prompts import (
    FunctionTagCustomToolGenerator,
    PromptTemplateGeneratorBase,
    PromptTemplate,
)
from llama_agent.utils.file_tree import list_files_in_repo
from llama_agent import REPO_DIR
from llama_agent.utils.ansi import red, yellow, magenta, blue, cyan
from subprocess import run
from textwrap import dedent
import textwrap
import difflib

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

def run_agent(
    client: LlamaStackClient,
    repo: str,
    problem_statement: str,
    sandbox_dir: Optional[str] = os.path.join(REPO_DIR, "sandbox"),
    eval_dir: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> Tuple[Literal["changes_made", "no_changes_made"], str, Optional[str]]:
    message = dedent("""\
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>\
    You are a software engineer tasked with solving a problem in a repo called {repo}. \
    Your problem is:

    <problem_statement>
    {problem_statement}
    </problem_statement>
    
    Please provide python code and bash commands to solve the problem. \
    We will execute your code in a sandbox environment.<|eot_id|>
    """).format(repo=repo, problem_statement=problem_statement)

    response = client.inference.completion(
        model_id=MODEL_ID,
        content=message,
    )
    print(response.content)
    pass