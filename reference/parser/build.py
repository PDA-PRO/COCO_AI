# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language
from ..config import AiConfig
import os

prev_path=""
if os.getcwd()!=AiConfig.parser_path:
    prev_path=os.getcwd()
    os.chdir(AiConfig.parser_path)

#토큰화에 필요한 tree_sitter 준비
Language.build_library(
# Store the library in the `build` directory
'./my-languages.so',
# Include one or more languages
[
    './tree-sitter-python/'
]
)

if prev_path:
  os.chdir(prev_path)