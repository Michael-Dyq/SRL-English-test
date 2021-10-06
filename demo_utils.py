from typing import Dict, Any

def create_token_char_offsets(text, tokens) -> Dict[str,Any]:
    char_offsets = []
    last_split_idx = -1
    for token in tokens:
        split_idx = text.find(token, last_split_idx+1)
        if split_idx == -1:
            split_idx = len(text)
        entry = {"form": token, "startCharOffset": split_idx, "endCharOffset": split_idx+len(token)}
        char_offsets.append(entry)
        last_split_idx = split_idx
        if last_split_idx == len(text):
            break
    return char_offsets

def create_sentence_view(tokens) -> Dict[str,Any]:
    sentence_view = {"viewName": "SENTENCE"}
    constituents = []
    sentence_end_positions = [i+1 for i,x in enumerate(tokens) if x=="."]
    sentence_end_positions = [0] + sentence_end_positions
    if len(tokens) not in sentence_end_positions:
        sentence_end_positions.append(len(tokens))
    constituents = [{"label": "SENTENCE", "score": 1.0, "start": sentence_end_positions[idx-1], "end": sentence_end_positions[idx]} for idx in range(1, len(sentence_end_positions))]
    view_data = [{"viewType": "", "viewName": "SENTENCE", "generator": "UserSpecified", "score": 1.0, "constituents": constituents}]
    sentence_view["viewData"] = view_data
    return sentence_view

def create_tokens_view(tokens) -> Dict[str,Any]:
    token_view = {"viewName": "TOKENS"}
    constituents = []
    for idx, token in enumerate(tokens):
        constituents.append({"label": token, "score": 1.0, "start": idx, "end": idx+1})
    view_data = [{"viewType": "", "viewName": "TOKENS", "generator": "UserSpecified", "score": 1.0, "constituents": constituents}]
    token_view["viewData"] = view_data
    return token_view
