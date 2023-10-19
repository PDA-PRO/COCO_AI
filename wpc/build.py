from tree_sitter import Language

def build_tree_sitter():
    #토큰화에 필요한 tree_sitter 준비
    Language.build_library(
    './reference/parser/my-languages.so',
    [
        './tree-sitter-python/'
    ]
    )

if __name__=="__main__":
   build_tree_sitter()