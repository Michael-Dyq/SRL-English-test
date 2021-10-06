#from nltk.stem import WordNetLemmatizer
import spacy
import numpy

import xml.etree.ElementTree as ET
import demo_utils
import os

class TabularView(object):
    def __init__(self, spacy_model):
        super().__init__()
        
        self.ta = {"corpusId": "", "id": ""}
        self.views = {}
        self.current_outputs = {}
        self.sp =spacy.load(spacy_model)
    
    def update_sentence(self, srl_output):
        generator = "srl_pipeline"
        tokens = srl_output["tokens"]
        text = " ".join(srl_output["words"])
        self.ta["text"] = text
        self.ta["tokens"] = tokens
        self.ta["tokenOffsets"] = demo_utils.create_token_char_offsets(text, tokens)
        sentence_end_positions = [i+1 for i,x in enumerate(tokens) if x=="."]
        if len(tokens) not in sentence_end_positions:
            sentence_end_positions.append(len(tokens))
        sentences = {"generator": generator, "score": 1.0, "sentenceEndPositions": sentence_end_positions}
        self.ta["sentences"] = sentences
        
        self.views = {}
        self.views["SENTENCE"] = demo_utils.create_sentence_view(tokens)
        self.views["TOKENS"] = demo_utils.create_tokens_view(tokens)
        self.ta["views"] = self.views.values()

    def update_view(self, view_name, srl_output):
        output = srl_output["predicates"]
        self.views[view_name] = self._create_srl_view(output, view_name)
        self.current_outputs[view_name] = output
        self.ta["views"] = list(self.views.values())

    def remove_view(self, view_name):
        if view_name in self.views:
            del self.views[view_name]
        if view_name in self.current_outputs:
            del self.current_outputs[view_name]

    def clear_table(self):
        self.views = {}
        self.ta = {"corpusId": "", "id": "", "text": "", "tokens": [], "tokenOffsets": [], "sentences": {}, "views": []}
        self.current_outputs = {}


    def get_textannotation(self):
        sanitized = self._sanitize(self.ta)
        # print(sanitized)
        # print(type(sanitized))
        return sanitized

    
    def _sanitize(self,x):
        if isinstance(x, (str, float, int, bool)):
            return x
        elif isinstance(x, numpy.ndarray):
            return x.tolist()
        elif isinstance(x, numpy.number):
            return x.item()
        elif isinstance(x, dict):
            return {key:self._sanitize(value) for key, value in x.items()}
        elif isinstance(x, numpy.bool_):
            return bool(x)
        elif isinstance(x, (list, tuple)):
            return [self._sanitize(x_i) for x_i in x]
        elif x is None:
            return "None"
        elif hasattr(x, "to_json"):
            return x.to_json()
        else:
            print(x, ' IS THE HARD ONE WE CANOT SANITIZE, IT IS OF TYPE, ', type(x))


    def _create_srl_view(self, frames, view_name):
        srl_view = {"viewName": view_name}
        constituents = []
        relations = []
        for frame in frames:
            predicate = frame["predicate"]
            tags = frame["tags"]
            predicate_idx = frame["predicate_index"]
            properties = {"SenseNumber": "NA", "predicate": predicate}
            constituent = {"label": "Predicate", "score": 1.0, "start": predicate_idx, "end": predicate_idx+1, "properties": properties}
            predicate_constituent_idx = len(constituents)
            constituents.append(constituent)
            active_tag = ""
            active_tag_start_idx = -1
            for tag_idx, tag in enumerate(tags):
                if tag in {"O", "B-V"}:
                    if active_tag != "":
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents)}
                        relations.append(relation)
                        constituents.append(constituent)
                        active_tag = ""
                        active_tag_start_idx = -1
                    continue
                if tag[2:] == active_tag:
                    continue
                else:
                    if active_tag != "":
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents)}
                        relations.append(relation)
                        constituents.append(constituent)
                    active_tag = tag[2:]
                    active_tag_start_idx = tag_idx
            # collect stragglers
            if active_tag != "":
                constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": len(tags)}
                relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents)}
                relations.append(relation)
                constituents.append(constituent)
        view_data = [{"viewType": "", "viewName": view_name, "generator": "multilingual_srl_pipeline", "score": 1.0, "constituents": constituents, "relations": relations}]
        srl_view["viewData"] = view_data
        return srl_view
    
