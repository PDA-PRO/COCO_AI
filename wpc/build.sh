#!/bin/bash

git clone https://github.com/tree-sitter/tree-sitter-python.git
python build.py
unzip desc/desc.zip -d desc