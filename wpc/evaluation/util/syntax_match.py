# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import tokenize
from io import StringIO

PY_LANGUAGE = Language(tspython.language())

def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    temp = []
    for x in out.split("\n"):
        if x.strip() != "":
            temp.append(x)
    return "\n".join(temp)

def corpus_syntax_match(references, candidates):
    parser = Parser(PY_LANGUAGE)
    match_count = 0
    match_count_candidate_to_reference = 0
    total_count = 0

    for i in range(len(candidates)):
        references_sample = references[i]
        candidate = candidates[i]
        for reference in references_sample:
            try:
                candidate = remove_comments_and_docstrings(candidate)
            except Exception:
                pass
            try:
                reference = remove_comments_and_docstrings(reference)
            except Exception:
                pass
            
            candidate_tree = parser.parse(bytes(candidate, "utf8")).root_node
            reference_tree = parser.parse(bytes(reference, "utf8")).root_node

            def get_all_sub_trees(root_node):
                node_stack = []
                sub_tree_sexp_list = []
                depth = 1
                node_stack.append([root_node, depth])
                while len(node_stack) != 0:
                    cur_node, cur_depth = node_stack.pop()
                    sub_tree_sexp_list.append([str(cur_node), cur_depth])
                    for child_node in cur_node.children:
                        if len(child_node.children) != 0:
                            depth = cur_depth + 1
                            node_stack.append([child_node, depth])
                return sub_tree_sexp_list

            cand_sexps = [x[0] for x in get_all_sub_trees(candidate_tree)]
            ref_sexps = [x[0] for x in get_all_sub_trees(reference_tree)]

            # TODO: fix, now we count number of reference subtrees matching candidate,
            #       but we should count number of candidate subtrees matching reference
            #       See (4) in "3.2 Syntactic AST Match" of https://arxiv.org/pdf/2009.10297.pdf
            for sub_tree in ref_sexps:
                if sub_tree in cand_sexps:
                    match_count += 1

            for sub_tree in cand_sexps:
                if sub_tree in ref_sexps:
                    match_count_candidate_to_reference += 1

            total_count += len(ref_sexps)
    # print(f'match_count       {match_count} / {total_count}')
    # print(f'match_count_fixed {match_count_candidate_to_reference} / {total_count}')
    score = match_count / total_count
    return score

def compute_syntax_match(references: Union[List[str], List[List[str]]], predictions: List[str]):
    references = [[x.strip() for x in ref] if isinstance(ref, list) else [ref.strip()] for ref in references]
    hypothesis = [x.strip() for x in predictions]

    return corpus_syntax_match(
        references, hypothesis
    )