import json 
from tqdm import tqdm
import pdb
from analysis.map_condition_phrases import read_embeddings
import numpy as np
import jsonlines
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import os
import sys
import random
import re
import scipy.cluster.hierarchy as hcluster
from shutil import copyfile
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pickle
import copy
import spacy
import time

sys.path.append("/data/rsg/nlp/darsh/pytorch-pretrained-BERT")

from nltk.tokenize import sent_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import matplotlib.pyplot as plt
from nltk.translate.meteor_score import meteor_score

from itertools import count, permutations
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from examples.run_fusion import InputExample
from examples.run_fusion import convert_examples_to_features
from allennlp.predictors.predictor import Predictor



torch.manual_seed(0)
random.seed(42)

lower_limit = 20
upper_limit = 25

#if len(sys.argv) > 1:
#    THRESHOLD = float(sys.argv[1])
#else:
THRESHOLD = 0.7

lemmatizer = WordNetLemmatizer()
spacy_nlp  = spacy.load("en_core_web_sm")

class PolicyChoices(nn.Module):
    def __init__(self, inputs):
        super(PolicyChoices, self).__init__()
        self.representation = nn.Sequential(
                nn.Linear(400,30),
                nn.Tanh(),
                nn.Linear(30,1)
                )
        self.affine1 = nn.Linear(inputs, 2*inputs)
        self.hidden  = nn.Linear(2*inputs, 2*inputs)
        self.affine2 = nn.Linear(2*inputs, 1)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = torch.cat((x[:,:2],\
                #x[:,202:204], x[:,-1].unsqueeze(1)),1)
                self.representation(torch.cat((x[:,2:202],x[:,2:202]),1)),\
                x[:,202:204], x[:,-1].unsqueeze(1)),1)
                #x[:,202:204]),1)
        x = self.affine1(x)
        x = torch.tanh(x)
        x = self.hidden(x)
        x = torch.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=0)



class Policy(nn.Module):
    def __init__(self, inputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(inputs, 8)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(8, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x).clamp(max=1e5)
        x = self.dropout(x).clamp(max=1e5)
        x = F.relu(x)
        action_scores = self.affine2(x).clamp(max=1e5)
        return F.softmax(action_scores, dim=1)

eps = np.finfo(np.float32).eps.item()
gamma=0.99


def select_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m     = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def select_action_choice(states, policy, epoch):
    states = torch.from_numpy(states).float()
    probs  = policy(states).squeeze()
    if len(probs.size()) == 0:
        action = 0
        policy.saved_log_probs.append(torch.log(probs.unsqueeze(0)))
        return 0
    if epoch < 0:
        n = Categorical(torch.cat((\
                torch.max(probs[:-1]).unsqueeze(0),probs[-1:]),0).squeeze())
        if n.sample().item() == 0:
            m      = Categorical(probs[:-1])
            action = m.sample()
        else:
            action = torch.tensor(len(probs)-1)

    else:
        m       = Categorical(probs)
        action = torch.argmax(probs)#m.sample()
    m      = Categorical(probs)
    policy.saved_log_probs.append(\
            m.log_prob(action).unsqueeze(0))
    return action.item()


def select_action_choice_pretrained(states, policy, fixed_choice):
    states = torch.from_numpy(states).float()
    probs  = policy(states).squeeze()
    if len(probs.size()) == 0:
        policy.saved_log_probs.append(torch.log(probs.unsqueeze(0)))
        return 0
    policy.saved_log_probs.append(torch.log(probs[fixed_choice]).unsqueeze(0))
    return fixed_choice


def finish_episode(policy, optimizer, prev_loss=[], batch_size=8, \
        do_policy=True):
    if prev_loss is None:
        prev_loss = []
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    #if len(returns) > 1:
    #    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * returns[0])
    
    policy_loss = torch.cat(policy_loss).sum()
    #print(policy_loss)
    #policy_loss = sum(policy.saved_log_probs) * sum(returns)
    if do_policy:
        prev_loss.append(policy_loss/len(returns))
    if len(prev_loss) == batch_size:
        optimizer.zero_grad()
        loss = 0
        for prev_l in prev_loss:
            if prev_l > 0 or True:
                loss += prev_l
        if loss > 0 or True:
            loss.backward()
            optimizer.step()
        #loss = sum(prev_loss)
        #loss.backward()
        #optimizer.step()
        prev_loss = []
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return prev_loss


class Environment:
    def __init__(self, gold_relations, gold_populations, embeddings, \
            cause_style_rationales_representations, sentence_representations,\
            all_gold_sentence_extracted_representations,\
            all_pubmed_sentence_representations,\
            rhs, gold_sentence,\
            pubmed_sentences=None, max_phrases=5):
        self.gold_relations = gold_relations
        self.gold_populations = gold_populations
        self.embeddings     = embeddings if embeddings is not []\
                else read_embeddings()
        self.cause_style_rationales_representations = \
                cause_style_rationales_representations
        self.sentence_representations = sentence_representations
        self.all_gold_sentence_extracted_representations = \
                all_gold_sentence_extracted_representations
        self.all_pubmed_sentence_representations = \
                all_pubmed_sentence_representations
        self.max_phrases    = max_phrases
        self.rhs            = rhs
        self.gold_sentence  = gold_sentence
        self.pubmed_sentences=pubmed_sentences


    def step(self, state, action, structures_selected, remaining_structures,\
            remaining_types, corresponding_sentiment, gold_sentiment):
        if action == 0 or len(remaining_structures)==0 or state[1]+1\
                >= self.max_phrases:
            state[1] += 1
            return state, self.get_reward(structures_selected), \
                    True, structures_selected
        else:
            if len(structures_selected) > 0:
                existing_reward = self.get_reward(structures_selected)
            else:
                existing_reward = 0.0
            best_structure  = None
            best_reward     = -1e10
            for additional,remaining_type in \
                    zip(remaining_structures,remaining_types):
                if remaining_type == 1:
                    new_reward  = self.get_reward(structures_selected + \
                            [additional]) - existing_reward
                else:
                    new_reward  = 0
                #assert new_reward >= 0
                if new_reward >= best_reward:
                    best_reward = new_reward
                    best_structure = additional
            state[0] += 1
            state[1] += 1
            assert best_structure is not None
            new_structures_selected = structures_selected + [best_structure]
            if state[1] >= self.max_phrases:
                import sys; sys.exit()
                return state, -1/self.max_phrases, True, new_structures_selected
                        #-1/self.max_phrases, True, \
                        #self.get_reward(new_structures_selected),\
                        #True, new_structures_selected
            return state, -1/self.max_phrases,\
                    len(new_structures_selected) >= self.max_phrases,\
                    new_structures_selected


    def get_reward(self, structures):

        overall_score = _best_overal_score(self.gold_relations,\
                [(x,y) for x,y,a,b,c,d in structures], self.embeddings)

        return overall_score


def get_relation_sentence_annotations(article_name, healthline_sentence,\
        metadata):

    summary_inputs = metadata[article_name]['summary_inputs']
    summary_annotations = summary_inputs\
            ['summary_healthline_relation_annotations'][healthline_sentence\
            .strip()]
    pubmed_inputs  = summary_inputs['summary_pubmed_articles']\
            [healthline_sentence]
    pubmed_annotations = {}
    pubmed_relation_sentiments = {}


    for pubmed in pubmed_inputs:
        if pubmed not in metadata['pubmed_sentences_annotations']:
            pubmed_annotations = {}
            break
        if 'pubmed_sentences_relation_annotations' in \
            metadata['pubmed_sentences_annotations'][pubmed]:
            pubmed_annotations[pubmed] = metadata[\
                    'pubmed_sentences_annotations'][pubmed]\
                    ['pubmed_sentences_relation_annotations']
            if 'relation_structure_sentiment' in \
                    metadata['pubmed_sentences_annotations'][pubmed]:
                pubmed_relation_sentiments[pubmed] = {}
                for element in metadata['pubmed_sentences_annotations'][pubmed]\
                        ['relation_structure_sentiment']:
                    if element[0][0].endswith("/") or element[0][0].endswith("\\"):
                        element[0][0] = element[0][0][:-1]
                    if element[0][1].endswith("/") or element[0][1].endswith("\\"):
                        element[0][1] = element[0][1][:-1]
                    pubmed_relation_sentiments[pubmed][tuple(element[0])] =\
                            element[2:]

            new_relations = []
            for p in pubmed_annotations[pubmed][0]:
                relation = []
                for q in p:
                    if q.endswith("/"):
                        q = q[:len(q)-1]
                    relation.append(q)
                new_relations.append(relation)
            new_contains = []
            for p in pubmed_annotations[pubmed][4]:
                contains = []
                for q in p:
                    if q.endswith("/") or q.endswith("\\"):
                        q = q[:len(q)-1]
                    contains.append(q)
                new_contains.append(contains)
            new_modifies = []
            for p in pubmed_annotations[pubmed][7]:
                modifies = []
                for q in p:
                    if q.endswith("/") or q.endswith("\\"):
                        q = q[:len(q)-1]
                    modifies.append(q)
                new_modifies.append(modifies)
            pubmed_annotations[pubmed][0] = new_relations
            pubmed_annotations[pubmed][4] = new_contains
            pubmed_annotations[pubmed][7] = new_modifies

    return summary_annotations, pubmed_annotations, pubmed_relation_sentiments


def get_entity_sentence_annotations(article_name, healthline_sentence, \
        metadata):

    summary_inputs = metadata[article_name]['summary_inputs']
    summary_annotations = {}
    summary_modifiers   = {}
    for sentence, tag in summary_inputs[\
            'summary_healthline_entity_annotations']:
        if sentence == healthline_sentence.strip():
            food_entities = _get_entities(sentence.split(),tag.split(),'Food')
            for food_entity in food_entities:
                summary_annotations[food_entity] = 'Food'
            condition_entities = _get_entities(sentence.split(),tag.split(),\
                    'Condition')
            for condition_entity in condition_entities:
                summary_annotations[condition_entity] = 'Condition'
            nutrition_entities = _get_entities(sentence.split(),tag.split(),\
                    'Nutrition')
            for nutrition_entity in nutrition_entities:
                summary_annotations[nutrition_entity] = 'Nutrition'
            population_entities = _get_entities(sentence.split(),tag.split(),\
                    'Population')
            for population_entity in population_entities:
                summary_modifiers[population_entity] = 'Population'
    pubmed_annotations = {}
    pubmed_modifiers   = {}
    pubmed_inputs  = summary_inputs['summary_pubmed_articles']\
        [healthline_sentence]
    for pubmed in pubmed_inputs:
        if pubmed not in metadata['pubmed_sentences_annotations']:
            pubmed_annotations = {}
            break
        pubmed_sentence_tuples = metadata['pubmed_sentences_annotations']\
                [pubmed]['pubmed_sentences_entity_annotations']
        for sentence,tags in pubmed_sentence_tuples:
            food_entities = _get_entities(sentence.split(),\
                    tags.split(),'Food')
            for entity in food_entities:
                pubmed_annotations[entity] = 'Food'
            condition_entities = _get_entities(sentence.split(),\
                    tags.split(),'Condition')
            for entity in condition_entities:
                pubmed_annotations[entity] = 'Condition'
            nutrition_entities = _get_entities(sentence.split(),\
                    tags.split(),'Nutrition')
            for entity in nutrition_entities:
                pubmed_annotations[entity] = 'Nutrition'
            population_entities = _get_entities(sentence.split(),\
                    tags.split(),'Population')
            for entity in population_entities:
                pubmed_modifiers[entity] = 'Population'
        pubmed_entities = list(pubmed_annotations.keys())
        for entity in pubmed_entities:
            if entity.endswith("/"):
                pubmed_annotations[entity[:entity.find("/")]] = \
                        pubmed_annotations[entity]
            if entity.endswith(","):
                pubmed_annotations[entity[:entity.find(",")]] = \
                pubmed_annotations[entity]
        summary_entities = list(summary_annotations.keys())
        for entity in summary_entities:
            if entity.endswith(","):
                summary_annotations[entity[:entity.find(",")]] = \
                        summary_annotations[entity]
        for entity in pubmed_modifiers:
            if entity.endswith("/"):
                pubmed_modifiers[entity[:entity.find("/")]] = \
                        pubmed_modifiers[entity]
            if entity.endswith(","):
                pubmed_modifiers[entity[:entity.find(",")]] = \
                        pubmed_modifiers[entity]
        for entity in summary_modifiers:
            if entity.endswith(","):
                summary_modifiers[entity[:entity.find(",")]] = \
                        summary_modifiers[entity]
    return summary_annotations, pubmed_annotations, \
            summary_modifiers, pubmed_modifiers


def paired_annotations(metadata):

    sentence_relation_annotations = {}
    sentence_entity_annotations   = {}
    sentence_modifier_entity_annotations = {}
    sentence_file_names = {}
    sentence_pubmed_relations_sentiments = {}

    for file_name in tqdm(metadata):
        if 'summary_inputs' not in metadata[file_name]:
            continue
        for healthline_sentence in tqdm(metadata[file_name]['summary_inputs']\
                ['summary_pubmed_articles']):
            summary,pubmed,relation_sentiments = \
                    get_relation_sentence_annotations(file_name,\
                    healthline_sentence, metadata)
            sentence_pubmed_relations_sentiments[healthline_sentence.strip()] =\
                    {}
            for pubmed_name in relation_sentiments:
                for relation in relation_sentiments[pubmed_name]:
                    sentence_pubmed_relations_sentiments[\
                            healthline_sentence.strip()][relation] = \
                            relation_sentiments[pubmed_name][relation]
            summary_entities,\
                    pubmed_entities,\
            summary_modifiers, pubmed_modifiers = \
                    get_entity_sentence_annotations(file_name, healthline_sentence,\
                    metadata)
            if len(pubmed) == 0:
                continue
            input = None
            for pubmed_file in pubmed:
                if input is None:
                    input = pubmed[pubmed_file]
                else:
                    for i in range(len(input)):
                        for x in pubmed[pubmed_file][i]:
                            input[i].append(x)
            sentence_relation_annotations[healthline_sentence.strip()] = \
                    [summary,list.copy(input)]
            sentence_entity_annotations[healthline_sentence.strip()] = \
                    [summary_entities,pubmed_entities]
            sentence_modifier_entity_annotations[healthline_sentence.strip()]\
                    = [summary_modifiers,pubmed_modifiers]
            sentence_file_names[healthline_sentence.strip()] = file_name

    return sentence_relation_annotations, sentence_file_names,\
            sentence_entity_annotations, sentence_modifier_entity_annotations,\
            sentence_pubmed_relations_sentiments


def _get_conjunction(fusion_model, tokenizer, sentence, label_list):
    examples = []
    examples.append(InputExample(guid=0, text_a=\
        sentence, text_b=None, label="", weight=0.0))
    eval_features = convert_examples_to_features(examples, \
            label_list, 128, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features],\
                        dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features],\
                        dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features],\
                        dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features],\
                        dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask,\
                        all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()
        tmp_eval_loss, logits = fusion_model(\
                input_ids, segment_ids, input_mask, label_ids)
        logits = list(logits.view(-1).detach().cpu().numpy())
        print(label_list[logits.index(max(logits))])
        return label_list[logits.index(max(logits))]
    return

def combine_individuals(sentences, fusion_model, tokenizer, label_list):
    if len(sentences) == 0:
        return ""
    final_sentence = sentences[0]
    for sentence in sentences[1:]:
        conjunction= _get_conjunction(fusion_model, tokenizer, \
                final_sentence + " ### " + sentence, label_list)
        if conjunction != "":
            final_sentence += " " + conjunction + " "+ sentence
        else:
            final_sentence += " " + sentence
    #final_sentence = final_sentence.replace("<X>","").replace("</X>","")\
    #        .replace("<Y>","").replace("</Y>","")
    #final_sentence = final_sentence.replace("  "," ")
    return final_sentence

def _compare_causes(input_cause, output_causes, embeddings, file_name):

    for output_cause in output_causes:
        if equal_strings(output_cause[0],input_cause[0],embeddings) and\
                equal_strings(output_cause[1],input_cause[1],embeddings):#\
            return 1.0
    return 0.0


def _compare_contains(input_contains, output_contains, embeddings, file_name):
    
    f = jsonlines.open("contains_correct.jsonl","a")
    for output_contain in output_contains:
        if equal_strings(output_contain[0],input_contains[0],embeddings)\
                and equal_strings(output_contain[1],input_contains[1],embeddings):
            dict = {'article':file_name,'input':output_contains,\
                    'output':input_contains}
            f.write(dict)
            f.close()
            return 1
    dict = {'article':file_name,'input':output_contains,\
            'output':input_contains}
    f = jsonlines.open("contains_wrong.jsonl","a")
    f.write(dict)
    f.close()
    return 0


def _best_overal_score(input_causes, output_causes, embeddings):

    selected_output_causes = {}
    total_score            = 0
    for input_cause in input_causes:
        current_score = 0
        current_selection = None
        for output_cause in output_causes:
            if selected_output_causes.get(tuple(output_cause),0) > 0:
                continue
            score = matching_score(input_cause[0].lower(),\
                    output_cause[0].lower(),embeddings)\
                    + matching_score(input_cause[1].lower(),\
                    output_cause[1].lower(),embeddings)
            score /= 2
            if score > current_score:
                current_score = score
                current_selection = tuple(output_cause)
        selected_output_causes[current_selection] = selected_output_causes.setdefault(\
                current_selection,0) + 1
        if current_score > 0:
            total_score += (current_score>=THRESHOLD)*current_score
    return total_score


def _best_causes_score(input_cause, output_causes, embeddings):

    best_score = -1
    for output_cause in output_causes:
        score = matching_score(input_cause[0],output_cause[0],embeddings)\
                + matching_score(input_cause[1],output_cause[1],embeddings)
        score /= 2.0
        if score > best_score:
            best_score = score
    return best_score


def _best_causes(input_cause, output_causes, embeddings):

    best_score = -1
    best_causes = None
    for output_cause in output_causes:
        score = matching_score(input_cause[0],output_cause[0],embeddings)\
                + matching_score(input_cause[1],output_cause[1],embeddings)
        if score > best_score:
            best_score = score
            best_causes = output_cause
    return best_causes


def _just_causes(input_cause, output_causes, embeddings):

    best_score = -1
    just_causes= None
    for output_cause in output_causes:
        score = matching_score(input_cause[1],output_cause[1],embeddings)
        if score > best_score:
            best_score = score
            just_causes= output_cause
    return just_causes

def _best_contains(input_contains, output_contains, embeddings):

    best_score = -1
    best_contains = None
    for output_contain in output_contains:
        score = matching_score(input_contains[0],output_contain[0],\
                embeddings) +\
                matching_score(input_contains[1],output_contain[1],\
                        embeddings)
        if score > best_score:
            best_score = score
            best_contains = output_contain
    return best_contains

def _get_entity_embeddings(string, embeddings):

    representation = [0.0] * 50
    word_list = []
    if ' ' in string:
        word_list = string.split()
    else:
        word_list = [string]
    for word in word_list:
        if word in embeddings:
            representation += embeddings[word]
    norm = np.linalg.norm(representation)
    if norm != 0.0:
        representation /= norm
    return np.array(representation)

def _get_phrase_embeddings(string, embeddings):
    return _get_entity_embeddings(string, embeddings)

def _get_all_entity_embeddings(entity_names, embeddings):

    entity_vectors = {}
    for entity in entity_names:
        entity_vectors[entity] = {}
        entity_vectors[entity]['names'] = []
        entity_vectors[entity]['vectors'] = []
        for phrase in entity_names[entity]:
            entity_vectors[entity]['names'].append(phrase)
            entity_vectors[entity]['vectors'].append(\
                    _get_entity_embeddings(phrase, embeddings))
        entity_vectors[entity]['vectors'] = \
                np.array(entity_vectors[entity]['vectors'])
    return entity_vectors


def equal_strings(string1, string2, embeddings):

    representation1 = [0.0] * 50
    representation2 = [0.0] * 50
    words1 = []
    words2 = []
    if ' ' in string1:
        words1 = string1.split()
    else:
        words1 = [string1]
    if ' ' in string2:
        words2 = string2.split()
    else:
        words2 = [string2]

    lemmatized_words1 = []
    lemmatized_words2 = []
    for word in words1:
        lemmatized_words1.append(lemmatizer.lemmatize(word))
    for word in words2:
        lemmatized_words2.append(lemmatizer.lemmatize(word))
    for word in lemmatized_words1:
        if word in lemmatized_words2:
            return 1

    representation1 = _get_entity_embeddings(string1, embeddings)
    representation2 = _get_entity_embeddings(string2, embeddings)

    if np.dot(representation1,representation2) > THRESHOLD:
        return 1
    return 0


def matching_score(string1, string2, embeddings):

    representation1 = _get_entity_embeddings(string1, embeddings)
    representation2 = _get_entity_embeddings(string2, embeddings)
    return np.dot(representation1,representation2)


def get_population_annotations(sentence_relation_annotations,\
        sentence_modifier_entity_annotations):

    sentence_population_relations = {}
    sentence_population_entities  = {}
    for sentence in sentence_relation_annotations:
        summary, pubmed = sentence_relation_annotations[sentence]
        summary_modifiers, pubmed_modifiers = \
                sentence_modifier_entity_annotations[sentence]
        summary_population_relations = []
        pubmed_population_relations  = []
        summary_population_entities  = []
        pubmed_population_entities   = []
        for r in summary[7]:
            #if r[0] in summary_modifiers and \
            #        summary_modifiers[r[0]] == 'Population':
            if not any([x in r[0] for x in ['male','female','men','women',\
                    'human','child','patient','individual','mammal','rat',\
                    'mice','animal']]):
                continue
            summary_population_relations.append(r)
            summary_population_entities.append(r[0])
        for r in pubmed[7]:
            #if r[0] in pubmed_modifiers and \
            #        pubmed_modifiers[r[0]] == 'Population':
            if not any([x in r[0] for x in ['male','female','men','women',\
                    'human','child','patient','individual','mammal','rat',\
                    'mice','animal']]):
                continue
            pubmed_population_relations.append(r)
            pubmed_population_entities.append(r[0])
        sentence_population_relations[sentence] = [\
                summary_population_relations, pubmed_population_relations]
        sentence_population_entities[sentence] = [\
                summary_population_entities, pubmed_population_entities]

    return sentence_population_relations, sentence_population_entities


def get_sentiment_annotations(metadata):

    sentence_sentiment = {}
    for file_name in metadata:
        if 'summary_inputs' not in metadata[file_name]:
            continue
        if 'summary_healthline_sentiment' not in \
            metadata[file_name]['summary_inputs']:
            continue
        for sentence, sentiment in metadata[file_name]['summary_inputs']\
                ['summary_healthline_sentiment'].items():
            sentence_sentiment[sentence] = sentiment

    return sentence_sentiment


def get_population_correlation(sentence_population_entities, \
        sentence_sentiment):

    non_human_neutral = 0
    non_human_total   = 0
    human_neutral     = 0
    human_total       = 0
    only_animal_total = 0
    only_animal_neutral = 0
    for sentence in sentence_population_entities:
        summary, pubmeds = sentence_population_entities[sentence]
        any_human = False
        only_animal = True
        for p in pubmeds:
            if any ([x in p for x in ['male','female','men','women',\
                    'human','child','patient','individual']]):
                any_human = True
                only_animal = False
                break
        if any_human:
            if sentence_sentiment[sentence] == 'Neutral':
                human_neutral += 1
            human_total       += 1
        else:
            if len(pubmeds) > 0:
                only_animal_total += 1
                if sentence_sentiment[sentence] == 'Neutral':
                    only_animal_neutral += 1
            if sentence_sentiment[sentence] == 'Neutral':
                non_human_neutral += 1
            non_human_total += 1

    print(non_human_neutral, \
            non_human_total, \
            only_animal_neutral, \
            only_animal_total, human_neutral, human_total)


def get_sentiment_statistics(sentence_sentiment, sentence_file_names,\
        metadata, splits):

    print(splits)
    sentiment_files = {}
    sentiment_sentences = {}
    for sentence,file_name in sentence_file_names.items():
        if metadata[file_name]['split'] not in splits:
            continue
        sentiment = sentence_sentiment[sentence]
        sentiment_files[sentiment] = \
                sentiment_files.setdefault(sentiment,[]) + \
                [file_name]
        sentiment_sentences[sentiment] = \
                sentiment_sentences.setdefault(sentiment,[]) +\
                [sentence]
        sentiment_files[sentiment] = list(set(sentiment_files[sentiment]))
        sentiment_sentences[sentiment] = \
                list(set(sentiment_sentences[sentiment]))
    print("Files " + str(sum([len(x) for x in sentiment_files.values()])) + " "+\
            str([[x,len(y)] for x,y in sentiment_files.items()]))

    print("Sentences " + str(sum([len(x) for x in sentiment_sentences.values()]))\
            + " "+ str([[x,len(y)] for x,y in sentiment_sentences.items()]))



def expand_pubmed_information(input_pubmed_information):

    pubmed_information = list.copy(input_pubmed_information)
    contains_information = pubmed_information[4]
    updated_information  = []
    entity_edges = {}
    entity_counts= {}
    for pair in contains_information:
        entity_edges[pair[0]] = set()
        entity_edges[pair[1]] = set()
        entity_counts[pair[0]]=0
        entity_counts[pair[1]]=0

    for pair in contains_information:
        entity_edges[pair[1]].add(pair[0])
        entity_counts[pair[1]] += 1
    
    for entity in sorted(entity_counts, key=entity_counts.get, reverse=True):
        stack = []
        visited = set()
        stack = list(entity_edges.get(entity,[]))
        candidates = set()
        while len(stack) - len(visited) > 0:
            current = stack[len(visited)]
            visited.add(current)
            parents = entity_edges.get(current, [])
            for parent in parents:
                if parent not in stack:
                    stack.append(parent)
                    candidates.add(parent)
        for candidate in candidates:
            pair = [candidate,entity]
            if pair not in updated_information+contains_information:
                updated_information.append(pair)
    output_pubmed_information = list.copy(pubmed_information)
    for pair in updated_information:
        output_pubmed_information[4].append(pair)
    current_relations = pubmed_information[0]
    updated_relations = []
    for relation in current_relations:
        for pair in output_pubmed_information[4]:
            if pair[1] == relation[0]:
                new_relation = [pair[0],relation[1],relation[2]]
                if new_relation not in updated_relations + current_relations:
                    updated_relations.append(new_relation)
    output_pubmed_information[4] = list.copy(pubmed_information[4]) + \
            list.copy(updated_relations)

    return pubmed_information


def _get_entities(tokens, labels, entity):

    entities_found = []
    current = ""
    for token,label in zip(tokens,labels):
        if label == 'B-'+entity:
            if current != "":
                entities_found.append(current)
            current = token
        elif label == 'I-'+entity:
            current += " " + token
        else:
            if current != "":
                entities_found.append(current)
            current = ""
    if current != "":
        entities_found.append(current)
    return entities_found


def get_corpus_entity_types(metadata):

    entity_types = {}
    for pubmed in metadata['pubmed_sentences_annotations']:
        if 'pubmed_sentences_entity_annotations' not in \
                metadata['pubmed_sentences_annotations'][pubmed]:
            continue
        for tokens, labels in \
                metadata['pubmed_sentences_annotations'][pubmed]\
                ['pubmed_sentences_entity_annotations']:
            food_entities = _get_entities(tokens.split(),labels.split(),'Food')
            for food in food_entities:
                entity_types[food] = entity_types.setdefault(food,[])+['Food']
            condition_entities = \
                    _get_entities(tokens.split(),labels.split(),'Condition')
            for condition in condition_entities:
                entity_types[condition] = \
                        entity_types.setdefault(condition,[])+['Condition']
            nutrition_entities = \
                    _get_entities(tokens.split(),labels.split(),'Nutrition')
            for nutrition in nutrition_entities:
                entity_types[nutrition] = \
                        entity_types.setdefault(nutrition,[])+['Nutrition']
    return entity_types

def get_title_entities(metadata):

    file_name_food = {}
    for file_name in metadata:
        file_name_food[file_name] = \
            _get_entities(metadata[file_name]['title_entities'][0].split(),\
            metadata[file_name]['title_entities'][1].split(), 'Food')\
            if 'title_entities' in metadata[file_name] else []
    return file_name_food


def add_title_contains(input, food_entities, embeddings):

    extra_contains = []
    for pair in input[4]:
        for food_entity in food_entities:
            if matching_score(food_entity,pair[0],embeddings) < 0.75:
                extra_contains.append([food_entity,pair[0]])
    output = list.copy(input)
    for pair in extra_contains:
        output[4].append(pair)
    return output


def get_target_entities(metadata, entities = ['Food','Condition','Nutrition']):

    entities_phrases = {}
    for file_name in metadata:
        if 'summary_inputs' not in metadata[file_name]:
            continue
        if 'summary_healthline_entity_annotations' not in \
                metadata[file_name]['summary_inputs']:
            continue
        for sentence in metadata[file_name]['summary_inputs']\
                ['summary_healthline_entity_annotations']:
            tokens = sentence[0].split()
            labels = sentence[1].split()
            assert len(tokens) == len(labels)
            for entity in entities:
                phrases = set(_get_entities(tokens, labels, entity))
                entities_phrases[entity] = \
                    entities_phrases.setdefault(entity,set()).union(phrases)

    return entities_phrases


def get_source_entities(metadata, entities = ['Food','Condition','Nutrition']):

    entities_phrases = {}
    ctr =  0
    for pubmed in metadata['pubmed_sentences_annotations']:
        if 'pubmed_sentences_entity_annotations' not in \
                metadata['pubmed_sentences_annotations'][pubmed]:
            continue
        ctr += 1
        for sentence in metadata\
            ['pubmed_sentences_annotations'][pubmed]\
            ['pubmed_sentences_entity_annotations']:
            tokens = sentence[0].split()
            labels = sentence[1].split()
            assert len(tokens) == len(labels)
            for entity in entities:
                phrases = set(_get_entities(tokens, labels, entity))
                entities_phrases[entity] = \
                    entities_phrases.setdefault(entity,set()).union(phrases)

    return entities_phrases



def get_best_entity(entity, target_type_embeddings, embeddings):

    embedding = _get_entity_embeddings(entity, embeddings).reshape(1,-1)
    dot_products = np.dot(embedding, target_type_embeddings.T)\
            .reshape(-1)
    return np.argmax(dot_products.reshape(-1))



def rewrite_from_target_dictionary(inputs, target_embeddings, embeddings):

    modified_inputs = list.copy(inputs)
    len1 = min(1000,len(inputs[0]))
    len2 = min(1000,len(inputs[4]))
    ctr = 0
    for i in range(len1):
        input = inputs[0][i]
        new_input = list.copy(input)
        new_input[0] = (target_embeddings['Food']['names']+\
                target_embeddings['Nutrition']['names'])\
                [get_best_entity(input[0], \
                np.concatenate((target_embeddings['Food']['vectors'], \
                target_embeddings['Nutrition']['vectors'])),\
                embeddings)]
        new_input[1] = target_embeddings['Condition']['names']\
                [get_best_entity(input[1], \
                target_embeddings['Condition']['vectors'], \
                embeddings)]
        modified_inputs[0].append(new_input)
    for i in range(len2):
        input = inputs[4][i]
        new_input = list.copy(input)
        new_input[0] = (target_embeddings['Food']['names']+\
                target_embeddings['Nutrition']['names'])\
                [get_best_entity(input[0], \
                np.concatenate((target_embeddings['Food']['vectors'], \
                target_embeddings['Nutrition']['vectors'])),\
                embeddings)]
        new_input[1] = target_embeddings['Nutrition']['names'][\
                get_best_entity(input[1],
                    target_embeddings['Nutrition']['vectors'],
                    embeddings)]
        modified_inputs[4].append(input)
    return modified_inputs


def rewrite_from_target_with_type_dictionary(inputs, type_dictionary, target_embeddings, embeddings):

    modified_inputs = list.copy(inputs)
    len1 = min(1000,len(inputs[0]))
    len2 = min(1000,len(inputs[4]))
    ctr = 0
    for i in range(len1):
        input = inputs[0][i]
        new_input = list.copy(input)
        if new_input[0] in type_dictionary:
            new_input[0] = target_embeddings[type_dictionary[new_input[0]]]['names']\
                    [get_best_entity(input[0], target_embeddings[type_dictionary[new_input[0]]]['vectors'],\
                                                                    embeddings)]
        else:
            new_input[0] = (target_embeddings['Food']['names']+\
                        target_embeddings['Nutrition']['names']+target_embeddings['Condition']['names'])\
                        [get_best_entity(input[0], \
                        np.concatenate((target_embeddings['Food']['vectors'], \
                        target_embeddings['Nutrition']['vectors'],\
                        target_embeddings['Condition']['vectors'])),\
                        embeddings)]
        new_input[1] = target_embeddings['Condition']['names']\
                [get_best_entity(input[1], \
                target_embeddings['Condition']['vectors'], \
                embeddings)]
        modified_inputs[0].append(new_input)
    for i in range(len2):
        input = inputs[4][i]
        new_input = list.copy(input)
        if new_input[0] in type_dictionary:
            new_input[0] = target_embeddings[type_dictionary[new_input[0]]]['names']\
                    [get_best_entity(input[0], target_embeddings[type_dictionary[new_input[0]]]['vectors'],\
                    embeddings)]
        else:
            new_input[0] = (target_embeddings['Food']['names']+\
                            target_embeddings['Nutrition']['names'])\
                            [get_best_entity(input[0], \
                                np.concatenate((target_embeddings['Food']['vectors'], \
                                target_embeddings['Nutrition']['vectors'])),\
                            embeddings)]
        new_input[1] = target_embeddings['Nutrition']['names'][\
                                get_best_entity(input[1],\
                                target_embeddings['Nutrition']['vectors'],\
                                embeddings)]
        modified_inputs[4].append(input)
    return modified_inputs



def get_causes_graph(causes_tuples, num_x=20, num_y=20, num_z=5):

    graph = np.ndarray(shape=(num_z), dtype=float)
    list_x = []
    list_y = []
    list_z = ['increases','decreases','controls','satisfies']
    for causes in causes_tuples:
        list_x.append(causes[0])
        list_y.append(causes[1])
    list_x = list(set(list_x))
    list_y = list(set(list_y))
    list_z = list(set(list_z))
    for ind,causes in enumerate(causes_tuples):
        z = list_z.index(causes[2])
        graph[z] += 1.0
    graph_norm = np.linalg.norm(graph.reshape(-1))
    if graph_norm > 0.0:
        return graph.reshape(-1)/graph_norm
    return graph.reshape(-1)


def get_causes_dict(causes, entity_types):

    dictionary = {}
    for causes_triplet in causes:
        if causes_triplet[2] not in dictionary:
            dictionary[causes_triplet[2]] = {}
        if causes_triplet[1] not in dictionary[causes_triplet[2]]:
            dictionary[causes_triplet[2]][causes_triplet[1]] = []
        if causes_triplet[0] not in entity_types:
            if causes_triplet[0].lower() in entity_types:
                causes_triplet[0] = causes_triplet[0].lower()
        if causes_triplet[0] not in entity_types and len(causes_triplet[0])>0:
            causes_triplet[0] = causes_triplet[0][0].upper() +\
                    causes_triplet[0][1:]
            if causes_triplet[0] not in entity_types:
                dictionary[causes_triplet[2]][causes_triplet[1]].append('Food')
        else:
            dictionary[causes_triplet[2]][causes_triplet[1]].append(\
                    entity_types.get(causes_triplet[0],'Food'))
    return dictionary


def get_named_causes_dict(causes):

    dictionary = {}
    for causes_triplet in causes:
        if causes_triplet[2] not in dictionary:
            dictionary[causes_triplet[2]] = {}
        if causes_triplet[1] not in dictionary[causes_triplet[2]]:
            dictionary[causes_triplet[2]][causes_triplet[1]] = []
        if causes_triplet[0] not in \
                dictionary[causes_triplet[2]][causes_triplet[1]]:
            dictionary[causes_triplet[2]][causes_triplet[1]].append(\
                    causes_triplet[0])
    return dictionary



def get_modified_sentence(sentence_annotations, sentence_entity_annotations,\
        sentence, causes, entities_type, current_sentence):

    output, _ = sentence_annotations[sentence]
    for contain in output[4]:
        output[0].append(contain + ['contains'])
    z_list = ['increases','decreases','controls','satisfies','contains']
    causes_type_dict = get_causes_dict(causes, \
            sentence_entity_annotations[sentence][0])
    get_original_type_dict = get_causes_dict(output[0], entities_type)
    causes_dict = get_named_causes_dict(causes)
    get_original_dict = get_named_causes_dict(output[0])
    modified_sentence = sentence.lower()
    selected_tuples = set()

    for cause in get_original_dict:
        if cause not in causes_dict:
            continue
        tuple_counts = {}
        for condition2 in causes_dict[cause].keys():
            for input2 in causes_dict[cause][condition2]:
                tuple_counts[tuple([condition2,input2])] = 0
        for condition1 in get_original_dict[cause].keys():
                for input1 in get_original_dict[cause][condition1]:
                    for cond2,in2 in sorted(tuple_counts, \
                            key=tuple_counts.get, reverse=False):

                        condition2 = cond2
                        input2     = in2
                        if input2 not in entities_type:
                            input2 = input2.lower()
                        if input1 not in sentence_entity_annotations\
                                [sentence][0]:
                            input1 = input1.lower()
                            if input1 not in sentence_entity_annotations\
                                    [sentence][0]:
                                pass
                        if tuple([condition1,input1]) \
                                in selected_tuples:
                            continue
                        if entities_type.get(input2,"Food") == \
                            sentence_entity_annotations[sentence][0].get(input1,"Food"):

                            modified_sentence = modified_sentence.replace(\
                                    input1.lower(),input2.lower())
                            modified_sentence = modified_sentence.replace(\
                                    condition1.lower(),condition2.lower())
                            tuple_counts[tuple([cond2,in2])] += 1
                            selected_tuples.add(tuple([condition1,input1]))
    return modified_sentence



def compare_annotations(sentence_annotations, sentence_entity_annotations,\
        embeddings, target_embeddings,\
        sentence_file_names, title_entities):

    causes_similar = []
    contains_similar=[]
    for sentence in tqdm(sentence_annotations):
        output, input = sentence_annotations[sentence]
        if len(input[0]) > 5000 or len(input[4]) > 5000:
            final_input = input
        else:
            new_input = add_title_contains(input, \
                title_entities[sentence_file_names[sentence]],\
                embeddings)
            final_input = expand_pubmed_information(new_input)
        input_entity_types = sentence_entity_annotations[sentence][1]
        #input = rewrite_from_target_dictionary(input, \
        #        target_embeddings, embeddings)
        #input = rewrite_from_target_with_type_dictionary(input, \
        #        input_entity_types, target_embeddings, embeddings)
        causes_scores = [_compare_causes(x, final_input[0], embeddings, \
                sentence_file_names[sentence]) for x in output[0]]
        contains_scores=[_compare_contains(x, final_input[4], embeddings,\
                sentence_file_names[sentence]) for x in output[4]]
        if len(causes_scores) > 0:
            causes_similar.append(sum(causes_scores)/(len(causes_scores)))
        if len(contains_scores) > 0:
            contains_similar.append(sum(contains_scores)/\
                    (len(contains_scores)))
    return causes_similar, contains_similar


def follow_up_annotations(sentence_annotations, embeddings, target_embeddings,\
        sentence_file_names, title_entities):

    sentence_causes = {}
    sentence_contains = {}
    sentence_all_causes = {}
    sentence_all_contains = {}
    gold_sentence_causes = {}
    gold_sentence_contains = {}

    for sentence in tqdm(sentence_annotations):
        output, input = list.copy(sentence_annotations[sentence])
        if len(input[0]) > 5000 or len(input[4]) > 5000:
            final_input = input
        else:
            new_input = add_title_contains(list.copy(input), \
                title_entities[sentence_file_names[sentence]],\
                embeddings)
            final_input = expand_pubmed_information(new_input)
        #current_causes = [_best_causes(x, final_input[0], embeddings) for x in \
        #        output[0]]
        #current_contains=[_best_contains(x, final_input[4], embeddings) for x in \
        #        output[4]]
        current_causes = [_just_causes(x, final_input[0], embeddings) for x in \
                output[0]]
        current_contains=[_just_causes(x, final_input[4], embeddings) for x in \
                output[4]]
        constraint_causes = [x for x in current_causes if x is not None]
        constraint_contains = [x for x in current_contains if x is not None]
        sentence_causes[sentence] = constraint_causes
        sentence_contains[sentence]= constraint_contains
        sentence_all_causes[sentence] = list.copy(final_input[0])
        sentence_all_contains[sentence] = list.copy(final_input[4])
        gold_sentence_causes[sentence] = list.copy(output[0])
        gold_sentence_contains[sentence]= list.copy(output[4])
    return sentence_causes, sentence_contains, sentence_all_causes,\
            sentence_all_contains, gold_sentence_causes, gold_sentence_contains


def get_property_rationales(metadata, sentence_relation_annotations,\
        sentence_file_names, sentence_entity_annotations,\
        sentence_relation_sentiments, splits=['train'], to_write=True):

    sentence_properties_annotations = {}
    for sentence in sentence_relation_annotations:
        sub_sentences = sent_tokenize(sentence)
        possible_sub_sentences = [1] * len(sub_sentences)
        if metadata[sentence_file_names[sentence]]['split'] not \
                in splits:
            continue
        pubmed, input = sentence_relation_annotations[sentence]
        sentence_properties_annotations[sentence] = {}
        for cause_triplet in pubmed[0]:
            x = cause_triplet[0]
            y = cause_triplet[1]
            if x.lower() not in sentence.lower() or y.lower() not in\
                    sentence.lower():
                continue
            for ind in range(len(sub_sentences)):
                if x.lower() in sub_sentences[ind].lower() or \
                        y.lower() in sub_sentences[ind].lower():
                    possible_sub_sentences[ind]=0
            template = sentence.lower()
            template = template.replace(x.lower(), "<X> "+x.lower()+" </X>")
            template = template.replace(y.lower(), "<Y> "+y.lower()+" </Y>")
            sentence_properties_annotations[sentence]['causes'] = \
                sentence_properties_annotations[sentence].setdefault('causes',\
                []) + [template]
        for contain_triplet in pubmed[4]:
            x = contain_triplet[0]
            y = contain_triplet[1]
            if x.lower() not in sentence.lower() or y.lower() not in\
                    sentence.lower():
                continue
            for ind in range(len(sub_sentences)):
                if x.lower() in sub_sentences[ind].lower()\
                        or y.lower() in sub_sentences[ind].lower():
                    possible_sub_sentences[ind] = 0
            template = sentence.lower()
            template = template.replace(x.lower(), "<X> "+x.lower()+" </X>")
            template = template.replace(y.lower(), "<Y> "+y.lower()+" </Y>")
            sentence_properties_annotations[sentence]['contains'] = \
                sentence_properties_annotations[sentence].setdefault('contains',\
                []) + [template]
        sentence_properties_annotations[sentence]['sentiment'] = [\
                sub_sentences[ind] for ind in range(len(sub_sentences))\
                if possible_sub_sentences[ind]==1]

    if to_write:
        text_files = {'causes':open("train_annotations_causes.txt","w"),\
                'contains':open("train_annotations_contains.txt","w"),\
                'sentiment':open("train_annotations_sentiment.txt","w")}
        json_files = {'causes':\
                jsonlines.open("train_annotations_causes.jsonl","w"),\
                'contains':\
                jsonlines.open("train_annotations_contains.jsonl","w"),\
                'sentiment':\
                jsonlines.open("train_annotations_sentiment.jsonl","w")}

        for sentence in sorted(sentence_properties_annotations):
            for property in sorted(sentence_properties_annotations[sentence]):
                for case in sorted(sentence_properties_annotations[sentence]\
                        [property]):
                    text_files[property].write("0\t0\t"+case+"\n")
                    dict = {'sentence':case,'original':sentence}
                    json_files[property].write(dict)

        for property in text_files.keys():
            text_files[property].close()
            json_files[property].close()


    return sentence_properties_annotations


def get_predicted_property_rationales():


    text_files = {'causes':open("train_annotations_causes.txt"\
            "","r").readlines()[1:],\
                'contains':open("train_annotations_contains.txt"\
                "","r").readlines()[1:],\
                'sentiment':open("train_annotations_sentiment.txt"\
                "","r").readlines()[1:]}
    json_files = {'causes':\
                jsonlines.open("train_annotations_causes.jsonl","r"),\
                'contains':\
                jsonlines.open("train_annotations_contains.jsonl","r"),\
                'sentiment':\
                jsonlines.open("train_annotations_sentiment.jsonl","r")}

    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/"\
            "allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

    sentence_properties_rationale_predictions = {}
    sentence_verb_phrase_rationale= {}
    sentence_constituency_parsing = {}
    sentence_verb_phrases = {}
    if os.path.exists("target_constituency_parse.p"):
        sentence_constituency_parsing = pickle.load(open(\
                "target_constituency_parse.p","rb"))
        sentence_verb_phrases = pickle.load(open(\
                "target_verb_phrases.p","rb"))
    for property in tqdm(['causes', 'contains', 'sentiment']):
        for r,line in tqdm(zip(json_files[property],text_files[property])):
            line = line.strip()
            sentence = r['original']
            if sentence not in sentence_properties_rationale_predictions:
                sentence_properties_rationale_predictions[sentence] = {}
                sentence_verb_phrase_rationale[sentence] = {}
            if property not in \
                    sentence_properties_rationale_predictions[sentence]:
                sentence_properties_rationale_predictions[sentence]\
                        [property] = []
                sentence_verb_phrase_rationale[sentence]\
                        [property] = []
            parts = line.split('\t')
            tokens= parts[2].split()
            take  = [1] * len(tokens)#[int(x) for x in parts[3].split()]
            span  = ""
            for token,t in zip(tokens,take):
                if t == 1:
                    span += token + " "
            span  = span.strip()
            #if (span in sentence_properties_rationale_predictions[sentence]\
            #        .get('causes',[]) or \
            #        sentence_properties_rationale_predictions[sentence]\
            #        .get('contains',[])) and property=='sentiment':
            #    continue
            if sentence not in sentence_constituency_parsing:
                sentence_constituency_parsing[sentence] = \
                    predictor.predict(sentence=sentence)
                stack = []
                if 'children' in sentence_constituency_parsing\
                        [sentence]['hierplane_tree']['root']:
                    stack += sentence_constituency_parsing[sentence]\
                            ['hierplane_tree']['root']['children']
                verb_phrases = []
                while len(stack) > 0:
                    child = stack[0]
                    if 'children' in child:
                        stack += child['children']
                    if child['nodeType'] == 'VP':
                        verb_phrases.append(child['word'])
                    del stack[0]
                sentence_verb_phrases[sentence] = verb_phrases
            tokened_sentence = ' '.join(tokens).strip()
            y_string = r['sentence'][r['sentence'].find("<Y>")+3:\
                    r['sentence'].find("</Y>")].strip()
            best_rationale = sentence
            for verb_phrase in sentence_verb_phrases[sentence]:
                if y_string not in verb_phrase:
                    continue
                if len(verb_phrase) < len(best_rationale):
                    best_rationale = verb_phrase
                    assert y_string in verb_phrase
            sentence_properties_rationale_predictions[sentence]\
                    [property].append(r['sentence'] + " ### " + best_rationale)
            sentence_verb_phrase_rationale[sentence][property]\
                    .append(best_rationale)


    pickle.dump(sentence_constituency_parsing, \
            open("target_constituency_parse.p","wb"))
    pickle.dump(sentence_verb_phrases, \
            open("target_verb_phrases.p","wb"))
    return sentence_properties_rationale_predictions
    #return sentence_verb_phrase_rationale

def get_fusion_training_data(sentence_extracted_rationales,\
        sentence_relation_annotations, sentence_file_names, title_entities, \
        embeddings):

    num_data_points    = 0

    all_title_entities = set()
    for entities in title_entities.values():
        for entity in entities:
            all_title_entities.add(entity)

    title_closest_entities = {}
    for entity in all_title_entities:
        current_entities = []
        current_scores   = []
        for entity2 in all_title_entities:
            if entity2 == entity:
                continue
            current_entities.append(entity2)
            current_scores.append(matching_score(entity, entity2, embeddings))
        indices = np.argsort(-np.array(current_scores))[:30]
        title_closest_entities[entity] = [current_entities[x] for x in indices]

    all_condition_entities = set()
    for sentence in sentence_relation_annotations:
        input,_ = sentence_relation_annotations[sentence]
        for cause in input[0]:
            all_condition_entities.add(cause[1])

    condition_closest_entities = {}
    for entity in all_condition_entities:
        current_entities = []
        current_scores   = []
        for entity2 in all_condition_entities:
            if entity2 == entity:
                continue
            current_entities.append(entity2)
            current_scores.append(matching_score(entity, entity2, embeddings))
        indices = np.argsort(-np.array(current_scores))[:200]
        condition_closest_entities[entity] = [current_entities[x] for x in \
                indices]


    training_jsonl = jsonlines.open("train_fusion.jsonl","w")
    training_instances = []
    for sentence in tqdm(sentence_extracted_rationales):
        if sentence_extracted_rationales[sentence].get('causes',[]) == []:
            continue
        sentence_causes = sentence_extracted_rationales[sentence]['causes']
        sentence_contain_relations = sentence_relation_annotations[sentence]\
                [0][4]
        sentence_conditions = [cause[cause.find("<Y>")+3:\
                cause.find("</Y>")].strip() for cause in sentence_causes]
        sentence_r_causes = [x[x.find("###")+3:].strip() for x in \
                sentence_causes]
        assert len(sentence_causes) == len(sentence_conditions)
        food_title = ""
        if len(title_entities[sentence_file_names[sentence]]) > 0:
            food_title  = title_entities[sentence_file_names[sentence]][0]\
                    .lower()
        for current_title in [food_title] + list(title_closest_entities.get(\
                food_title,[])):
            possible_sentences = sent_tokenize(sentence.lower())
            useful_sentences = []
            for possible_sentence in possible_sentences:
                tokenized_possible_sentence = ' '.join([\
                        token.text for token in \
                        spacy_nlp(possible_sentence.lower())]).strip()
                for cause_sentence in sentence_causes:
                    cause_sentence = cause_sentence[\
                            cause_sentence.find("###")+3:].strip()
                    if cause_sentence.lower() in tokenized_possible_sentence:
                        useful_sentences.append(tokenized_possible_sentence)
                        break
            target_sentence = ' '.join(useful_sentences).strip()
            if target_sentence.strip() == "":
                continue
            if food_title in target_sentence and len(food_title) > 0:
                target_sentence = target_sentence.replace(food_title,\
                        current_title)
            source_sentence = current_title + " ### "
            ignore_data = False
            for contain_relation in sentence_contain_relations:
                if contain_relation[0].lower() in target_sentence:
                    target_sentence     = target_sentence.replace(\
                            contain_relation[0].lower(), current_title.lower())
                    contain_relation[0] = current_title
                else:
                    ignore_data = True
                    break
                source_sentence += ' contains '.join(contain_relation) + " ### "
            if ignore_data:
                break
            reduced_sentence_causes = []
            #if len(sentence_contain_relations) > 0:
            meta_source_sentence = source_sentence
            short_sentence_causes = list(set([sentence_cause[sentence_cause.find("###")\
                    +3:].strip() for sentence_cause in sentence_causes if \
                    sentence_cause[sentence_cause.find("###")+3:].strip()\
                    != sentence]))
            for sent in short_sentence_causes:
                reduced_sentence_causes.append(sent)
                for sen in short_sentence_causes:
                    if sen == sent:
                        continue
                    if sent in sen:
                        del reduced_sentence_causes[-1]
                        break
            if any([sent.lower() not in target_sentence for sent in \
                    reduced_sentence_causes]):
                continue
            cause_indices = [target_sentence.index(sent.lower()) for sent in \
                    reduced_sentence_causes]
            sorted_indices= np.argsort(np.array(cause_indices))
            reduced_sentence_causes = [reduced_sentence_causes[ind] for ind in\
                    sorted_indices]
            reduced_sentence_conditions = [sentence_conditions[sentence_r_causes\
                    .index(sentence_cause)] for sentence_cause in \
                    reduced_sentence_causes]
            for ind, (cause_sentence, cause_condition) in enumerate(\
                    zip(reduced_sentence_causes,\
                    reduced_sentence_conditions)):
                possible_replacements = [cause_condition] + \
                        condition_closest_entities.get(cause_condition, [])
                for possible_replacement in possible_replacements:
                    source_sentence = meta_source_sentence
                    target_sentence = ' '.join(useful_sentences).strip()
                    for i, (c_s, c_c) in enumerate(\
                            zip(reduced_sentence_causes,\
                            reduced_sentence_conditions)):
                        if i == ind:
                            c_s = c_s.replace(cause_condition, \
                                       possible_replacement)
                            target_sentence = target_sentence.replace(\
                                    cause_condition,\
                                    possible_replacement)
                            source_sentence += c_s + " ### "
                        else:
                            source_sentence += c_s + " ### "
                    fraction = sum([word in source_sentence for word\
                            in target_sentence.split()])/\
                            len(target_sentence.split())
                    if not( fraction >= 0.8 or (fraction >= 0.5 and \
                            'contains' in meta_source_sentence) ):
                        continue
                    fusion_dict = {'target': target_sentence}
                    fusion_dict['source'] = source_sentence
                    training_instances.append(fusion_dict)
                    num_data_points += 1
    print("Created %d training data points" %num_data_points)
    random.shuffle(training_instances)
    for instance in training_instances:
        training_jsonl.write(instance)
    training_jsonl.close()


def get_mapped_cosine_similarities(metadata, sentence_extracted_rationales,\
        sentence_file_names, sentence_all_causes, embeddings):

    vectorizer = TfidfVectorizer(max_features=5000)
    corpus_sentences = []
    sentence_pubmed_articles = {}
    file_name_pubmed = {}
    pubmed_sentences = []
    sentence_pubmeds = {}
    for file_name in metadata:
        if 'summary_inputs' not in metadata[file_name]:
            continue
        if 'summary_pubmed_articles' not in metadata[file_name]\
                ['summary_inputs']:
            continue
        for sentence,pubmeds in metadata[file_name]\
                ['summary_inputs']['summary_pubmed_articles'].items():
            corpus_sentences += [sentence]
            sentence_pubmed_articles[sentence.strip()] = pubmeds
        file_name_pubmed[file_name] = []
        if 'pubmed_sentences' not in metadata[file_name]:
            continue
        for pubmed in metadata[file_name]["pubmed_sentences"]:
            sentence_pubmeds[pubmed] = []
            for pubmed_text,label_token in metadata[file_name]\
                    ["pubmed_sentences"][pubmed][1:]:
                corpus_sentences.append(' '.join(pubmed_text).strip())
                file_name_pubmed[file_name].append(corpus_sentences[-1])
                sentence_pubmeds[pubmed].append(' '.join(pubmed_text).strip())

    f_train = jsonlines.open("importance_train.jsonl","w")
    f_dev   = jsonlines.open("importance_dev.jsonl","w")

    f_train_rationale = {'causes':open('T5_train_general.txt','w'),\
            'sentiment':open('T5_train_sentiment.txt','w')}
    f_dev_rationale = {'causes':open('T5_dev_general.txt','w'),\
            'sentiment':open('T5_dev_sentiment.txt','w')}
    f_train_rationale_jsonl = {'causes':jsonlines.open('T5_train_general.jsonl','w'),\
            'sentiment':jsonlines.open('T5_train_sentiment.jsonl','w')}
    f_dev_rationale_jsonl   = {'causes':jsonlines.open('T5_dev_general.jsonl','w'),\
            'sentiment':jsonlines.open('T5_dev_sentiment.jsonl','w')}

    sentence_pubmed_causes = {}
    sentence_pubmed_causes_importance = {}
    for sentence in sentence_extracted_rationales:
        for pubmed in sentence_pubmed_articles[sentence]:
            if pubmed not in sentence_pubmed_causes:
                sentence_pubmed_causes[pubmed] = []
                sentence_pubmed_causes_importance[pubmed] = []
            for pubmed_sentence in sentence_pubmeds[pubmed]:
                for cause_triplet in sentence_all_causes[sentence]:
                    if cause_triplet[0] in pubmed_sentence and \
                            cause_triplet[1] in pubmed_sentence:
                        modified_sentence = pubmed_sentence.replace(\
                                cause_triplet[0], " <X> " + cause_triplet[0]\
                                + " </X> ")
                        modified_sentence = modified_sentence.replace(\
                                cause_triplet[1], " <Y> " + cause_triplet[1]\
                                + " </Y> ")
                        if modified_sentence not in sentence_pubmed_causes[pubmed]:
                            sentence_pubmed_causes[pubmed].append(modified_sentence)
                            sentence_pubmed_causes_importance[pubmed].append(0.0)



    if args.run_features:
        line_importance_file = jsonlines.open("T4_dev_general.jsonl","w")
    input_dicts = []
    for pubmed in sentence_pubmed_causes:
        for pubmed_sentence in sentence_pubmed_causes[pubmed]:
            dict = {'sentence':pubmed_sentence, 'gold_label': 'increases',\
                    'uid': 0, 'pubmed': pubmed}
            input_dicts.append(dict)
            if args.run_features:
                line_importance_file.write(dict)
    if args.run_features:
        line_importance_file.close()
    if args.run_features:
        os.chdir("/data/rsg/nlp/darsh/pytorch-pretrained-BERT")
        os.system("python examples/run_causes.py --task_name re_task --do_eval --do_lower_case --data_dir /data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir t4_general_causes_output --output_preds")
        os.chdir("/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT")
        copyfile("/data/rsg/nlp/darsh/"\
                "pytorch-pretrained-BERT/t4_general_causes_output/"\
                "preds.jsonl","pubmed_line_importance_preds.jsonl")
    prediction_file = jsonlines.open(\
            "pubmed_line_importance_preds.jsonl","r")
    for p,input_dict in zip(prediction_file,input_dicts):
        pubmed_file = input_dict['pubmed']
        sentence    = input_dict['sentence']
        pred_prob   = float(p['increases'])
        sentence_pubmed_causes_importance[pubmed_file][\
                sentence_pubmed_causes[pubmed_file].index(sentence)] = \
                pred_prob

    prediction_file.close()

    healthline_sentence_pubmed_causes  = {}
    healthline_sentence_pubmed_causes_importance = {}
    for sentence in sentence_extracted_rationales:
        healthline_sentence_pubmed_causes[sentence] = []
        healthline_sentence_pubmed_causes_importance[sentence] = []
        for pubmed in sentence_pubmed_articles[sentence]:
            healthline_sentence_pubmed_causes[sentence] += \
                sentence_pubmed_causes[pubmed]
            healthline_sentence_pubmed_causes_importance[sentence] += \
                sentence_pubmed_causes_importance[pubmed]


    vectorizer.fit_transform(corpus_sentences)
    for sentence in tqdm(sentence_extracted_rationales):
        for property in sentence_extracted_rationales[sentence]:
            if property in ['contains']:
                continue
            accepted_indices = []
            for case in tqdm(sentence_extracted_rationales[sentence]\
                    .get(property,[])):
                sent_representation = vectorizer.transform([case])
                pubmed_sentences    = []
                pubmed_sentences_entities = {}
                for pubmed in sentence_pubmed_articles[sentence]:
                    if pubmed in sentence_pubmeds:
                        pubmed_sentences += sentence_pubmeds[pubmed]
                pubmed_representations = vectorizer.transform(pubmed_sentences)
                dot_products = np.dot(sent_representation, \
                        pubmed_representations.transpose()).toarray().flatten()
                if property == 'causes':
                    for pubmed_sentence in pubmed_sentences:
                        for cause in sentence_all_causes.get(sentence,[]):
                            if cause[0] in pubmed_sentence and cause[1] in \
                                pubmed_sentence:
                                pubmed_sentences_entities[pubmed_sentence] = \
                                pubmed_sentences_entities.setdefault(pubmed_sentence,\
                                []) + [cause]

                    entity_sums  = []
                    x_string     = sentence[sentence.find("<X>"):\
                            sentence.find("</X>")]
                    y_string     = sentence[sentence.find("<Y>"):\
                            sentence.find("</Y>")]

                    enlarged_sentences = []
                    enlarged_original  = []
                    full_pubmed_representations = \
                            pubmed_representations.toarray()
                    for pubmed_sentence, pubmed_representation in \
                            tqdm(zip(pubmed_sentences, \
                            full_pubmed_representations)):
                        max_score = 0.0
                        for cause in pubmed_sentences_entities.get(pubmed_sentence,[]):
                            modified_sentence = pubmed_sentence.replace(\
                                    cause[0], ' <X> ' + cause[0] + ' </X> ')
                            modified_sentence = modified_sentence.replace(\
                                    cause[1], ' <Y> ' + cause[1] + ' </Y> ')
                            existing_general_prob = 0.1
                            if sentence in healthline_sentence_pubmed_causes:
                                if modified_sentence in \
                                        healthline_sentence_pubmed_causes\
                                        [sentence]:
                                    existing_general_prob = \
                                        healthline_sentence_pubmed_causes_importance[sentence]\
                                        [healthline_sentence_pubmed_causes[sentence].index(modified_sentence)]
                            enlarged_original.append(pubmed_sentence)
                            enlarged_sentences.append(modified_sentence)
                            current_score = matching_score(cause[0],\
                                    x_string, embeddings) + \
                                    matching_score(cause[1],\
                                    y_string, embeddings) * existing_general_prob
                            entity_sums.append(current_score)
                            if current_score > max_score:
                                max_score = current_score
                    dot_products = np.array(entity_sums)
                    pubmed_sentences = enlarged_sentences
                    if len(enlarged_original) == 0:
                        continue
                    pubmed_representations = vectorizer.transform(enlarged_original)
                dot_products += np.dot(sent_representation, \
                        pubmed_representations.transpose()).toarray().flatten()
                accepted_indices.append(np.argmax(dot_products))


            accepted_indices = list(set(accepted_indices))
            for ind,p_sentence in enumerate(pubmed_sentences):
                label = 'increases' if ind in accepted_indices else 'NA'
                if label == 'NA':
                    if random.choice([i for i in range(15)]) != 0:
                        continue
                dict = {'sentence':p_sentence, 'gold_label':label, 'uid':0}
                if random.choice([0,0,0,0,1]) == 1:
                    f_dev.write(dict)
                    f_dev_rationale_jsonl[property].write(dict)
                    f_dev_rationale[property].write(str(ind)+"\t"+\
                        str(int(label=='increases'))+"\t"+p_sentence+"\n")
                else:
                    f_train.write(dict)
                    f_train_rationale_jsonl[property].write(dict)
                    f_train_rationale[property].write(str(ind)+"\t"+\
                        str(int(label=='increases'))+"\t"+p_sentence+"\n")
    f_dev.close()
    f_train.close()

    for property in f_train_rationale:
        f_train_rationale[property].close()
        f_train_rationale_jsonl[property].close()
        f_dev_rationale[property].close()
        f_dev_rationale_jsonl[property].close()

    return vectorizer

             


def match_dicts(input_dict, output_dict):

    total = 0
    for r in output_dict:
        for c in output_dict[r]:
            total += len(output_dict[r][c])
    if total == 0:
        return 0
    total2 = 0
    for r in input_dict:
        for c in input_dict[r]:
            total2 += len(input_dict[r][c])
    if total2 == 0:
        return 0

    matching = 0
    for r in output_dict:
        if r not in input_dict:
            continue
        types1 = sorted([output_dict[r][c] for c in output_dict[r]])
        types2 = sorted([input_dict[r][c] for c in input_dict[r]])

        indices_selected = set()
        for xs in types1:
            best_ind = -1
            best_len = 0
            for ind,ys in enumerate(types2):
                if ind in indices_selected:
                    continue
                ctr  = 0
                set_xs = set(xs)
                for v in set_xs:
                    ctr += min(xs.count(v), ys.count(v))
                len1  = ctr
                ctr = 0
                if len1 > best_len:
                    best_len = len1
                    best_ind = ind
            if best_ind != -1:
                indices_selected.add(best_ind)
                matching += best_len

    assert matching <= total
    assert matching <= total2
    return (matching/total + matching/total2)/2
    #return (matching/total - 100*((total-matching)/total))


def create_importance_classification_data(sentence_file_names, metadata, \
        sentence_causes, sentence_all_causes, sentence_contains,\
        sentence_all_contains, split_name):
    train_data = jsonlines.open("importance_train.jsonl","w")
    dev_data   = jsonlines.open("importance_dev.jsonl","w")
    test_data  = jsonlines.open("importance_test.jsonl","w")
    dictionary = {'train':train_data, 'dev':dev_data}
    for sentence in sentence_causes:
        file_name = sentence_file_names[sentence]
        split     = metadata[file_name]["split"]
        if split in ['train','dev']:
            candidates = sentence_all_causes[sentence] + sentence_all_contains[sentence]
            random.shuffle(candidates)
            for s1,s2 in zip(sentence_causes[sentence]+sentence_contains[sentence],\
                    candidates):
                dict1 = {'sentence':s1[0]+" # "+s1[1],'gold_label':'increases'}
                dictionary[split].write(dict1)
                if s2 != s1:
                    dict2 = {'sentence':s2[0]+" # "+s2[1],'gold_label':'NA'}
                    dictionary[split].write(dict2)
        elif split == split_name:
            for s1 in sentence_causes[sentence]+sentence_contains[sentence]:
                dict1 = {'sentence':s1[0]+" # "+s1[1],'gold_label':'increases',\
                        'original_sentence':sentence,'structure':s1}
                test_data.write(dict1)
                if file_name == '11-proven-benefits-of-bananas':
                    if s1[0].lower() == "salmon":
                        assert False
            for s2 in sentence_all_causes[sentence]+sentence_all_contains[sentence]:
                if s2 not in sentence_causes[sentence]+sentence_contains[sentence]:
                    dict2 = {'sentence':s2[0]+" # "+s2[1],'gold_label':'NA',\
                            'original_sentence':sentence,'structure':s2}
                    test_data.write(dict2)
                    if file_name == '11-proven-benefits-of-bananas':
                        if s1[0].lower() == "salmon":
                            assert False
    train_data.close()
    dev_data.close()
    test_data.close()


def get_predicted_structures(input_file, predicted_file, embeddings,\
        sentence_file_names, title_entities,\
        sentence_relation_sentiments, metadata, cluster_threshold=1.5,\
        prob_threshold=0.2):

    sentence_structures = {}
    sentence_probabilities={}
    reader1 = jsonlines.open(input_file,"r")
    reader2 = jsonlines.open(predicted_file,"r")

    for r1,r2 in tqdm(zip(reader1,reader2)):
        #if not (1 - float(r2['NA']) > prob_threshold) and \:
        #    continue
        causes_structures = []
        contains_structures=[]
        causes_representations = []
        contains_representations=[]
        if r2['pred_label'] != 'NA' or True:
            sentence_structures[r1['original_sentence'].strip()] = \
                    sentence_structures.setdefault(\
                    r1['original_sentence'].strip(),[])\
                    + [r1['structure']]
            sentence_probabilities[r1['original_sentence'].strip()] = \
                    sentence_probabilities.setdefault(\
                    r1['original_sentence'].strip(),[])\
                    + [1-float(r2['NA'])]

    for sentence in tqdm(sentence_structures):
        causes_structures = []
        contains_structures=[]
        causes_representations = []
        contains_representations=[]
        causes_probabilities = []
        contains_probabilities = []
        causes_sentiments= []
        for structure,prob in zip(sentence_structures[sentence],\
                sentence_probabilities[sentence]):
            if len(title_entities[sentence_file_names[sentence]]) > 0:
                rep1 = _get_entity_embeddings(structure[0], embeddings)
                rep2 = _get_entity_embeddings(title_entities[\
                    sentence_file_names[sentence]][0], embeddings)
                if np.dot(rep1,rep2) > 0.95:
                    prob = 1.0
            if prob < prob_threshold:
                continue
            if len(structure) == 2:
                contains_structures.append(structure)
                rep1 = _get_entity_embeddings(structure[0], embeddings)
                rep2 = _get_entity_embeddings(structure[1], embeddings)
                contains_representations.append(np.concatenate((rep1,rep2)))
                contains_representations[-1] = \
                        np.concatenate((contains_representations[-1],[0,0,0]))
                contains_probabilities.append(prob)
            else:
                causes_structures.append(structure)
                rep1 = _get_entity_embeddings(structure[0], embeddings)
                rep2 = _get_entity_embeddings(structure[1], embeddings)
                causes_representations.append(np.concatenate((rep1,rep2)))
                causes_probabilities.append(prob)
                causes_sentiments.append(\
                    sentence_relation_sentiments[sentence]\
                    .get(tuple(structure),['Good'])[0])
                if causes_sentiments[-1] == 'Bad':
                    causes_probabilities[-1] += 100.0 * \
                            float(sentence_relation_sentiments\
                            [sentence][tuple(structure)][1])
                if tuple(structure) in sentence_relation_sentiments[sentence]\
                    and sentence_relation_sentiments[sentence]\
                        [tuple(structure)][-1] == True:
                    causes_probabilities[-1] *= 1.0
                sentiment_feature = [0,0,0]
                sentiment_feature[['Good','Bad','Neutral']\
                        .index(causes_sentiments[-1])] = 1
                causes_representations[-1] = \
                        np.concatenate((causes_representations[-1],\
                        sentiment_feature))
        chain_structures = []
        chain_representations = []
        chain_probabilities = []
        chain_sentiments = []
        for i in range(len(contains_structures)):
            rep1 = _get_entity_embeddings(contains_structures[i][0], embeddings)
            rep2 = _get_entity_embeddings(contains_structures[i][1], embeddings)
            if np.dot(rep1,rep2) > 0.75:
                continue
            if [contains_structures[i]] not in chain_structures:
                chain_structures.append([contains_structures[i]])
                chain_probabilities.append(contains_probabilities[i])
                chain_representations.append(contains_representations[i])
            for j in range(len(causes_structures)):
                if [causes_structures[j]] not in chain_structures:
                    chain_structures.append([causes_structures[j]])
                    chain_probabilities.append(causes_probabilities[j])
                    chain_representations.append(causes_representations[j])
                rep1 = _get_entity_embeddings(contains_structures[i][1],\
                        embeddings)
                rep2 = _get_entity_embeddings(causes_structures[j][0],\
                        embeddings)
                if np.dot(rep1,rep2) > 0.95:
                    chain_rep = (contains_representations[i] +\
                            causes_representations[j])/2
                    chain_prob= contains_probabilities[i] + \
                            causes_probabilities[j]
                    chain_structures.append([contains_structures[i],\
                            causes_structures[j]])
                    chain_representations.append(chain_rep)
                    chain_probabilities.append(chain_prob)

        if len(contains_structures) == 0:
            chain_structures = [[x] for x in causes_structures]
            chain_representations = list.copy(causes_representations)
            chain_probabilities = list.copy(causes_probabilities)


        chain_clusters = []
        if len(chain_representations) > 1:
            chain_clusters = hcluster.fclusterdata(chain_representations,\
                    cluster_threshold, criterion="distance").tolist()

            chain_structures_dict = {}
            if chain_clusters != []:
                for ind, cluster_index in enumerate(chain_clusters):
                    if cluster_index not in chain_structures_dict:
                        chain_structures_dict[cluster_index] = ind
                    else:
                        if chain_probabilities[ind] > \
                                chain_probabilities[chain_structures_dict[cluster_index]]:
                            chain_structures_dict[cluster_index] = ind
                sentence_structures[sentence] = []
                for cluster_index,ind in chain_structures_dict.items():
                    assert len(chain_structures[ind]) <= 2
                    for structure in chain_structures[ind]:
                        sentence_structures[sentence].append(structure)
            if sentence_file_names[sentence] == 'legumes-good-or-bad':
                #pass
                if ['raw red kidney beans', 'acute gastroenteritis', 'increases']\
                    in sentence_structures[sentence]:
                        print("Our friend is found")
        else:
            assert len(chain_representations) <= 1
            sentence_structures[sentence] = []
            for chain_structure in chain_structures:
                for structure in chain_structure:
                    sentence_structures[sentence].append(structure)


    return sentence_structures


def get_causes_contains_structures(sentence_structures, sentence_all_causes,\
        sentence_all_contains):

    sentence_learnt_causes = {}
    sentence_learnt_contains = {}
    for sentence in sentence_structures:
        sentence_learnt_causes[sentence] = []
        sentence_learnt_contains[sentence]= []
        for structure in sentence_structures[sentence]:
            if len(structure) == 3:
                sentence_learnt_causes[sentence] = sentence_learnt_causes\
                        .setdefault(sentence,[]) + [structure]
            else:
                sentence_learnt_contains[sentence] = sentence_learnt_contains\
                        .setdefault(sentence,[]) + [structure]

    return sentence_learnt_causes, sentence_learnt_contains


def predict_importance_sentences(metadata, split_name, args):

    input_file = jsonlines.open("importance_dev.jsonl","w")
    sentiment_file = jsonlines.open("dev_sentiment.jsonl","w")
    all_pubmeds = {}
    pubmed_sentences = {}
    pubmed_sentences_labels = {}
    pubmed_file_name = {}

    for file_name in metadata:
        if 'split' not in metadata[file_name]:
            continue
        if metadata[file_name]['split'] != split_name:
            continue
        if 'summary_inputs' not in metadata[file_name]:
            continue
        if 'summary_pubmed_articles' not in metadata[file_name]\
                ['summary_inputs']:
            continue
        for sentence,pubmeds in metadata[file_name]['summary_inputs']\
                ['summary_pubmed_articles'].items():
            for pubmed in pubmeds:
                all_pubmeds[file_name] = all_pubmeds.setdefault(file_name,[])+\
                        [pubmed]
                pubmed_file_name[pubmed] = file_name

    for file_name in all_pubmeds:
        for pubmed in all_pubmeds[file_name]:
            if pubmed not in pubmed_sentences:
                pubmed_sentences[pubmed] = []
                if pubmed in metadata\
                        ['pubmed_sentences_annotations']:
                    title_sentence = metadata['pubmed_sentences_annotations']\
                            [pubmed]['pubmed_sentences_entity_annotations'][0]\
                            [0]
                    for sentence,labels in metadata\
                        ['pubmed_sentences_annotations'][pubmed]\
                        ['pubmed_sentences_entity_annotations'][1:]:
                        if sentence in pubmed_sentences[pubmed] or \
                                    sentence == title_sentence:
                            continue
                        pubmed_sentences[pubmed].append(sentence)
                        pubmed_sentences_labels[sentence] = labels

    input_sentences = []
    sentiment_sentences = []
    for pubmed in pubmed_sentences:
        for sentence in pubmed_sentences[pubmed]:
            dict = {'sentence':sentence, 'pubmed':pubmed, 'uid':0, \
                    'gold_label':'NA', 'file_name':pubmed_file_name[pubmed],\
                    'entity_string':pubmed_sentences_labels[sentence]}
            sentiment_dict = {'sentence':sentence, 'pubmed':pubmed, 'uid':0, \
                    'gold_label':'Good'}
            sentiment_sentences.append(sentiment_dict)
            input_file.write(dict)
            input_sentences.append(dict)
            sentiment_file.write(sentiment_dict)
    input_file.close()
    sentiment_file.close()

    # get sentence sentiments
    if args.run_features:
        from evaluation.create_sentiment_outputs import produce_sentiment_outputs
        results = produce_sentiment_outputs("dev_sentiment.jsonl", \
                "full_sentiment_classification")
        copyfile("/data/rsg/nlp/darsh/"\
                "pytorch-pretrained-BERT/full_sentiment_classification/"\
                "preds.jsonl","pubmed_"+split_name+"_preds.jsonl")
    else:
        #results = jsonlines.open("/data/rsg/nlp/darsh/"\
        #    "pytorch-pretrained-BERT/full_sentiment_classification/"\
        #    "preds.jsonl","r")
        results = jsonlines.open("pubmed_"+split_name+"_preds.jsonl")
   
    # get_importance_probabilities
    if args.run_features:
        os.chdir("/data/rsg/nlp/darsh/pytorch-pretrained-BERT")
        os.system("python examples/run_importance.py --task_name re_task --do_eval --do_lower_case --data_dir /data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir Importance_Classification --output_preds")
        os.chdir("/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT")
        copyfile("/data/rsg/nlp/darsh/"\
                "pytorch-pretrained-BERT/Importance_Classification/"\
                "preds.jsonl","pubmed_importance_"+split_name+"_preds.jsonl")
    prediction_file = jsonlines.open(\
            "pubmed_importance_"+split_name+"_preds.jsonl","r")

    output_sentences= []
    for p,r in zip(prediction_file,results):
        p['sentiment'] = r['pred_label']
        output_sentences.append(p)
    prediction_file.close()

    print("Sending %d pubmed sentences for downstream fun" \
            %len(input_sentences))
    return input_sentences, output_sentences



def _get_baseline_sentences(env, sentences):

    kept_sentences = set()
    attended_relations = set()
    for g_r in env.gold_relations:
        if _compare_causes(g_r,list(kept_sentences),env.embeddings,""):
            continue
        for sentence in sentences:
            if _compare_causes(g_r,[sentence],env.embeddings,""):
                kept_sentences.add(tuple(sentence))
                break
    return list(kept_sentences), 0, 0



def _get_policy_sentences(env, sentences, policy, optimizer, prev_loss=[],\
        batch_size=8):

    rewards = []
    for I in range(1):
        state = np.array([0,0])
        selected_sentences = []
        for t in range(env.max_phrases):
            remaining_sentences = [x for x in sentences if x not \
                in selected_sentences]
            action= select_action(state, policy)
            state, reward, done, selected_sentences = env.step(\
                state, action, selected_sentences, remaining_sentences)
            policy.rewards.append(reward)
            if done:
                rewards.append(sum(policy.rewards))
                break
        prev_loss = \
                finish_episode(policy, optimizer, prev_loss, batch_size)
    return selected_sentences, (sum(rewards)/len(rewards)), prev_loss



def _get_threshold_policy_sentences(env, sentences, importances, policy, \
        optimizer, prev_loss=[], batch_size=8, threshold=0.4):

    rewards = []
    for I in range(1):
        state = np.array([0,0,1.0,1.0])
        selected_sentences = []
        for t in range(env.max_phrases):
            if len(selected_sentences) != 0:
                remaining_sentences = [x for x in sentences if \
                    not _compare_causes(x,selected_sentences,env.embeddings,"")]
            else:
                remaining_sentences = sentences
            remaining_importances = [y for x,y in zip(\
                    sentences,importances) if x in remaining_sentences]
            assert len(remaining_sentences) == len(remaining_importances)
            if len(remaining_sentences) > 0:
                max_importance = max(remaining_importances)
                max_ind        = remaining_importances.index(max_importance)
                state[2]       = max_importance
                action         = select_action(state, policy)
                state, reward, done, selected_sentences = env.step(\
                    state, action, selected_sentences, [remaining_sentences\
                    [max_ind]])
                policy.rewards.append(reward)
            else:
                done           = True
            if done:
                rewards.append(sum(policy.rewards))
                break
        prev_loss = \
                finish_episode(policy, optimizer, prev_loss, batch_size)
    return selected_sentences, (sum(rewards)/len(rewards)), prev_loss
            

    

def _get_choice_policy_sentences(env, sentences, importances, sentiments, types,\
        sentence_representations, policy, optimizer, gold_sentiment, vectorizer, \
        cause_style_rationales_representations,\
        prev_loss=[], batch_size=8, epoch_number=-1, pretrain_until=-1, \
        repeat_instance=5):

    try:
        embeddings
    except:
        embeddings = read_embeddings()
    do_pretrain = False#epoch_number < pretrain_until

    rewards = []
    sentence_full_representations = []
    for sentence, sentence_representation in zip(sentences,\
            sentence_representations):
        sentence_full_representations.append(\
                list(_get_entity_embeddings(sentence[0],embeddings))+\
                list(_get_entity_embeddings(sentence[1],embeddings))+\
                list(_get_entity_embeddings(sentence[0],embeddings))+\
                list(_get_entity_embeddings(sentence[1],embeddings)))
                #list(_get_entity_embeddings(sentence[5],embeddings))+\
                #list(sentence_representation))

    sentences_index = {}
    for ind,sent in enumerate(sentences):
        sentences_index[tuple(sent[:3]+sent[4:])] = ind

    for I in range(repeat_instance):
        #probs = [env.get_reward([sentence]) for sentence in sentences]
        selected_sentences = []
        selected_representations = []
        selected_full_representations = []
        for t in range(env.max_phrases):
            #remaining_sentences = [x for x in sentences if x not in \
            #        selected_sentences]
            remaining_sentences = []
            for i,x in enumerate(sentences):
                found=False
                for s_s in selected_sentences:
                    found = True
                    for a,b in zip(x,s_s):
                        if ((type(a)!=np.ndarray and a!=b) or \
                        (type(a)==np.ndarray and not \
                        all([a1==b1 for a1,b1 in zip(a,b)]))):
                            found = False
                            break
                    if found:
                        break
                if found:
                    continue
                remaining_sentences.append(x)

            remaining_importances=[]
            remaining_sentiments =[]
            remaining_types      = []
            remaining_representations=[]
            remaining_counts     = []
            for ind in range(len(sentences)):
                found = False
                for r_s in remaining_sentences:
                    found=True
                    for a,b in zip(sentences[ind],r_s):
                        if ((type(a)!=np.ndarray and a!=b) or \
                        (type(a)==np.ndarray and not \
                        all([a1==b1 for a1,b1 in zip(a,b)]))):
                            found=False
                            break
                    if found:
                        break
                if found:
                    remaining_importances.append(importances[ind])
                    remaining_sentiments.append(sentiments[ind])
                    remaining_types.append(types[ind])
                    remaining_representations.append(\
                            list(_get_entity_embeddings(sentences[ind][0],embeddings))+\
                            list(_get_entity_embeddings(sentences[ind][1],embeddings))+\
                            list(_get_entity_embeddings(sentences[ind][0],embeddings))+\
                            list(_get_entity_embeddings(sentences[ind][1],embeddings)))
                            #list(_get_entity_embeddings(sentences[ind][5],embeddings))+\
                            #list(sentence_representations[ind]))
                    remaining_counts.append(env.pubmed_sentences[sentences[ind][-2]].lower()\
                            .count(sentences[ind][0].lower())+\
                            env.pubmed_sentences[sentences[ind][-2]].lower()\
                            .count(sentences[ind][1].lower())+\
                            5*env.pubmed_sentences[sentences[ind][-2]].lower()\
                            [:env.pubmed_sentences[sentences[ind][-2]].lower().find("###")]\
                            .count(sentences[ind][0].lower())+\
                            5*env.pubmed_sentences[sentences[ind][-2]].lower()\
                            [:env.pubmed_sentences[sentences[ind][-2]].lower().find("###")]\
                            .count(sentences[ind][1].lower()))
            #remaining_importances=[y for x,y in zip(sentences,importances)\
            #        if x in remaining_sentences]
            #remaining_sentiments =[y for x,y in zip(sentences,sentiments)\
            #        if x in remaining_sentences]
            #remaining_types      =[y for x,y in zip(sentences,types)\
            #        if x in remaining_sentences]
            #remaining_representations = [list(_get_entity_embeddings(x[0],embeddings))+\
            #        list(_get_entity_embeddings(x[1],embeddings))+list(y) for x,y in zip(sentences,\
            #        sentence_representations) if x in remaining_sentences]
            assert len(remaining_sentences) == len(remaining_importances)
            minimum_differences= [1] * len(remaining_sentences)
            min_text_differences=[1] * len(remaining_sentences)
            overall_similarity  =[0] * len(remaining_sentences)
            like_target         =[0] * len(remaining_sentences)
            #for remaining_sentence,remaining_type,remaining_representation\
            #        in zip(remaining_sentences,remaining_types,\
            #        remaining_representations):
            #    min_difference = 0
            #    text_difference=0
            #    if remaining_type == 0:
            #        min_difference = 0.5
            #        text_difference = 0.5
            #    else:
            #        assert len(selected_sentences) == len(selected_representations)
            #        for sentence,sentence_representation in \
            #                zip(selected_sentences,selected_representations):
            #            difference = (matching_score(remaining_sentence[0],\
            #                sentence[0],env.embeddings) + \
            #                matching_score(remaining_sentence[1],\
            #                sentence[1],env.embeddings))/2
            #            sentence_difference = np.dot(sentence_representation,\
            #                    remaining_representation)
            #            #difference = (difference+sentence_difference)/2 
            #            if difference > min_difference:
            #                min_difference = difference
            #            if sentence_difference > text_difference:
            #                text_difference = sentence_difference
            #    minimum_differences.append(min_difference)
            #    min_text_differences.append(text_difference)
            if len(selected_representations) > 0 and \
                    len(remaining_representations) > 0:
                sent_remaining_representations = [x[100:150] for x in \
                        remaining_representations]
                max_similarity_with_selected = \
                    np.max(np.dot(np.array(sent_remaining_representations),\
                    np.array(selected_representations).transpose()),axis=1)
                minimum_differences = [1 - x for x in max_similarity_with_selected]
                min_text_differences =[1 - x for x in max_similarity_with_selected]
            words_selected = sum([len(structure[2][\
                        structure[2].find("###")+3:].strip().split()) \
                            for structure in selected_sentences])
            all_lhs=[]
            if len(remaining_representations)>0:
                sent_remaining_representations = [x[100:150] for x in \
                        remaining_representations]
                if env.cause_style_rationales_representations\
                        != []:
                    all_lhs= np.sum(np.dot(np.array(sent_remaining_representations),\
                    env.cause_style_rationales_representations.transpose()),axis=1)
                    all_max_lhs = np.max(np.dot(np.array(sent_remaining_representations),\
                            env.cause_style_rationales_representations.transpose()),axis=1)
                else:
                    all_lhs = np.sum(np.dot(np.array(sent_remaining_representations),\
                            np.array([[0]*50]).transpose()),axis=1)
                    all_max_lhs = np.max(np.dot(np.array(sent_remaining_representations),\
                            np.array([[0]*50]).transpose()),axis=1)
            if len(selected_full_representations) == 0:
                selected_full_representations = [[0] * 200]
            selected_full_representations = np.array(\
                    selected_full_representations)
            selected_full_representations = np.sum(selected_full_representations,\
                    axis=0)
            states =[[len(selected_sentences),t] + \
                    #importance +\
                    representation + [int(1-min_diff<=0.9)]\
                    #+[int(lhs>env.rhs)]\
                    +[0]\
                    +list(selected_full_representations)\
                    +[remaining_count]\
                    #+ [min_diff] + [int(sim<=0.9)]\
                    #+  sentiment\
                    #+ [type]\ +
                    #+ [l_t]\
                    #+[int(words_selected<=lower_limit)] + [int(words_selected+\
                    #len(remaining_sentence[2][\
                    #remaining_sentence[2].find("###")+3:].strip()\
                    #.split()) < upper_limit)] + \
                    #[len(selected_sentences) > 1]
                    for remaining_sentence, importance,\
                            representation, min_diff,\
                    min_text_diff, sentiment, type, sim, l_t, \
                    remaining_count in zip(remaining_sentences,\
                    remaining_importances, remaining_representations,\
                    minimum_differences,\
                    min_text_differences,remaining_sentiments,\
                    remaining_types, overall_similarity, like_target,\
                    remaining_counts)\
                    ]# + \
            if t > 2 or len(states)==0:
                states += \
                    [[len(selected_sentences),t]+\
                    [0]*200 +[-1,-1] + [0]*200+[0]]\
                    +[[len(selected_sentences),t]+\
                    [0]*200 + [-2,-2] + [0]*200+[0]]
            assert len(remaining_sentences) == len(states)-2 or t<=2
            if do_pretrain:
                fixed_choice = len(states)-1
                original_reward = env.get_reward(selected_sentences)
                best_reward = 0.0
                for i in range(len(states)-1):
                    assert i < len(remaining_sentences)
                    new_reward = env.get_reward(selected_sentences + \
                        [remaining_sentences[i]])
                    if new_reward-original_reward> best_reward:
                        best_reward = new_reward-original_reward
                        fixed_choice= i

                action = select_action_choice_pretrained(np.array(states), policy,\
                        fixed_choice)
                target_label = torch.LongTensor([fixed_choice])
                criterion = nn.CrossEntropyLoss()
                prev_loss.append(criterion(\
                        policy(torch.from_numpy(\
                        np.array(states)).float()).unsqueeze(0), \
                        target_label.unsqueeze(0)))
            else:
                action = select_action_choice(np.array(states), policy, t)
            corresponding_action = action < len(remaining_sentences) or t==0
            corresponding_extra  = []
            corresponding_sentiment = []
            corresponding_type = []
            if corresponding_action:
                corresponding_extra = remaining_sentences[action]
                corresponding_sentiment = remaining_sentiments[action]
                corresponding_type = remaining_types[action]
            if action == len(states)-1:
                n_selected_sentences = []
                for selected_sentence in selected_sentences:
                    n_selected_sentence = list(selected_sentence)
                    n_selected_sentence[-1] = ''
                    n_selected_sentences.append(tuple(n_selected_sentence))
            else:
                n_selected_sentences = selected_sentences
            _, reward, done, n_selected_sentences = env.step(\
                    [len(selected_sentences),t,-1,-1,-1], \
                    corresponding_action, n_selected_sentences,\
                    [corresponding_extra], [corresponding_type],\
                    corresponding_sentiment, \
                    gold_sentiment)
            if len(selected_sentences) == len(n_selected_sentences):
                pass
            else:
                selected_sentences += [n_selected_sentences[-1]]
            selected_representations = []
            selected_full_representations = []
            for selected_sentence in selected_sentences:
                selected_full_representations.append(\
                    sentence_full_representations[\
                    sentences_index[tuple(selected_sentence[:3]+\
                    selected_sentence[4:])]])
                selected_representations.append(\
                        sentence_representations[\
                        sentences_index[tuple(selected_sentence[:3]+\
                        selected_sentence[4:])]])
            policy.rewards.append(reward)
            if done:
                if t==0:
                    pass
                break

        env_reward = env.get_reward(selected_sentences)
        rewards.append(sum(policy.rewards))
        prev_loss = \
                finish_episode(policy, optimizer, prev_loss, batch_size,\
                not do_pretrain)

    return selected_sentences, (sum(rewards)/len(rewards)), prev_loss


def _get_clustered_sentences(sentences, sentence_probs,\
        vectorizer, cluster_threshold=1.0):

    all_representations = vectorizer.transform(sentences)
    all_clusters = hcluster.fclusterdata(all_representations.toarray(),\
            cluster_threshold, criterion="distance").tolist()
    cluster_sentences = {}
    cluster_probabilities = {}
    assert len(all_clusters) == len(sentences)
    for ind,cluster in enumerate(all_clusters):
        cluster_sentences[cluster] = cluster_sentences.setdefault(cluster,\
                []) + [sentences[ind]]
        if type(sentence_probs[ind]) == float:
            cluster_probabilities[cluster] = \
                    cluster_probabilities.setdefault(\
                    cluster,[]) + [sentence_probs[ind]]
        else:
             cluster_probabilities[cluster] = \
                     cluster_probabilities.setdefault(\
                cluster,[]) + [sentence_probs[ind][0]]
    selected_cluster_sentences = {}
    for cluster in cluster_sentences:
        max_i = cluster_probabilities[cluster].index(\
                max(cluster_probabilities[cluster]))
        selected_cluster_sentences[cluster] = cluster_sentences[cluster][max_i]
    return list(selected_cluster_sentences.values())


def get_causes_contains_from_pubmed(input_sentences, output_sentences,\
        sentence_all_causes, sentence_all_contains, \
        sentence_file_names, metadata, sentence_sentiment, \
        vectorizer, embeddings, gold_sentence_causes, policy,\
        optimizer, args, sentence_extracted_rationales,\
        fusion_model, tokenizer, label_list,\
        pubmed_entity_types,\
        property_style_rationales,\
        cluster_threshold=1.0, split_name='test'):


    #cause_style_rationales_representations = \
    #    np.array([_get_entity_embeddings(sentence, embeddings)\
    #    for sentence in property_style_rationales['causes']])
    property_style_rationales_representations = {}
    for property in property_style_rationales:
        property_style_rationales_representations[property] = \
        np.array([_get_entity_embeddings(sentence, embeddings)\
        for sentence in property_style_rationales[property]])\

    sentence_extracted_rationale_representations = {}
    for sentence in sentence_extracted_rationales:
        for property,spans in sentence_extracted_rationales[sentence].items():
            if len(spans) == 0:
                continue
            reps = np.array([_get_entity_embeddings(span, embeddings) for span \
                    in spans])
            if sentence not in sentence_extracted_rationale_representations:
                sentence_extracted_rationale_representations[sentence] = reps
            else:
                sentence_extracted_rationale_representations[sentence] = \
                        np.concatenate((\
                        sentence_extracted_rationale_representations[sentence]\
                        , reps), axis=0)

    all_gold_sentence_extracted_representations = []
    for file_name in metadata:
        if 'summary_inputs' not in metadata[file_name]:
            continue
        if 'summary_healthline_entity_annotations' not in \
                metadata[file_name]['summary_inputs']:
            continue
        for sentence,labels in \
                metadata[file_name]['summary_inputs']\
                ['summary_healthline_entity_annotations']:
            all_gold_sentence_extracted_representations.append(\
                    _get_entity_embeddings(sentence, embeddings))
    all_gold_sentence_extracted_representations = np.array(\
            all_gold_sentence_extracted_representations)

    all_pubmed_sentence_representations = []
    all_pubmed_rationale_representations= []
    pubmed_sentence_verb_phrases        = pickle.load(open(\
            "pubmed_sentence_verb_phrases.p","rb"))
    for pubmed in metadata['pubmed_sentences_annotations']:
        if 'pubmed_sentences_entity_annotations' not in \
                metadata['pubmed_sentences_annotations'][pubmed]:
            continue
        for sentence,label in metadata['pubmed_sentences_annotations']\
                [pubmed]['pubmed_sentences_entity_annotations']:
            all_pubmed_sentence_representations.append(\
                    _get_entity_embeddings(sentence, embeddings))
            verb_phrases = pubmed_sentence_verb_phrases.get(sentence,[])
            for verb_phrase in verb_phrases:
                all_pubmed_rationale_representations.append(\
                        _get_entity_embeddings(verb_phrase,embeddings))
    all_pubmed_sentence_representations = np.array(\
            all_pubmed_sentence_representations)
    all_pubmed_rationale_representations= np.array(\
            all_pubmed_rationale_representations)
    rhs                                 =\
            np.sum(np.dot(all_pubmed_rationale_representations,\
            (np.concatenate((property_style_rationales_representations['causes'],\
            property_style_rationales_representations['contains'],\
            property_style_rationales_representations['sentiment']))).transpose()))\
            /len(all_pubmed_rationale_representations)

    gold_sentiment_file = jsonlines.open("dev_sentiment.jsonl","w")
    gold_sentence_sentiments = {}
    gold_sentence_inputs= []
    for sentence in sentence_all_causes:
        dict = {'sentence':sentence,'uid':0,'gold_label':'Good'}
        gold_sentence_inputs.append(dict)
        gold_sentiment_file.write(dict)
    gold_sentiment_file.close()

    if args.run_features:
        from evaluation.create_sentiment_outputs import produce_sentiment_outputs
        results = produce_sentiment_outputs("dev_sentiment.jsonl", \
                                       "full_sentiment_classification")
        copyfile("/data/rsg/nlp/darsh/"\
                "pytorch-pretrained-BERT/full_sentiment_classification/"\
                "preds.jsonl","healthline_sentiment.jsonl")
    else:
        results = jsonlines.open("healthline_sentiment.jsonl","r")
    for gold_input,result in zip(gold_sentence_inputs,results):
        gold_sentence_sentiments[gold_input['sentence']] = result['pred_label']


    all_cause_tuples = set()
    all_cause_sentence_tuples = {}
    for sentence,causes in sentence_all_causes.items():
        if sentence not in all_cause_sentence_tuples:
            all_cause_sentence_tuples[sentence] = set()
        for cause in causes:
            all_cause_sentence_tuples[sentence].add(tuple(cause[:2]))
            all_cause_tuples.add(tuple(cause[:2]))

    pubmed_important_sentences = {}
    all_important_sentences = {}
    all_important_sentences_entities = {}
    all_important_sentences_importance_probabilities = {}
    all_important_sentences_pubmed = {}
    number_of_sentences = 0
    pubmed_all_sentences_considered = set()
    for input,output in zip(input_sentences, output_sentences):
        if output['pred_label'] != "NA" or True:
            pubmed = input['pubmed']
            sentence=input['sentence']
            pubmed_all_sentences_considered.add(sentence)
            label_str=input['entity_string']
            all_important_sentences_importance_probabilities[sentence] = \
                    float(output['increases'])
            all_important_sentences_pubmed[sentence] = pubmed
            if pubmed not in pubmed_important_sentences:
                pubmed_important_sentences[pubmed] = {}
            if output['sentiment'] not in pubmed_important_sentences[pubmed]:
                pubmed_important_sentences[pubmed][output['sentiment']] = []
            if sentence not in \
                    pubmed_important_sentences[pubmed][output['sentiment']]:
                pubmed_important_sentences[pubmed][output['sentiment']]\
                    .append(sentence)
                all_important_sentences[sentence] = label_str
                all_important_sentences_entities[sentence] = {}
                for entity_type in ['Food','Nutrition','Condition']:
                    all_important_sentences_entities[sentence]\
                        [entity_type] = _get_entities(sentence.split(),\
                        label_str.split(), entity_type)
                number_of_sentences += 1


    sentence_pubmeds = {}
    sentence_structures = {}
    to_annotate_sentence_original = {}

    relation_sentences_considered = set()

    if not os.path.exists(\
            split_name + "_importance_pubmed_general.txt"\
            ".rationale.machine_readable.tsv")\
            or not \
            os.path.exists(\
            split_name + "_importance_pubmed_sentiment."\
            "txt.rationale.machine_readable.tsv") or True:

        f_causes = open(split_name + "_importance_pubmed_general.txt","w")
        f_sentiment = open(split_name + "_importance_pubmed_sentiment.txt","w")

        pubmed_sentence_verb_phrases = \
                pickle.load(open("pubmed_sentence_verb_phrases.p","rb"))

        for sentence in all_important_sentences_entities:
            f_sentiment.write("0\t0\t"+sentence+"\n")
            #if sentence.startswith("This could explain"):
            #"This could explain in part why the severe deficiency in <X> omega-3 </X> intake pointed by numerous epidemiologic studies may increase the <Y> brain </Y> 's vulnerability representing an important risk factor in the development and/or deterioration of certain cardio- and neuropathologies ."
            pubmed = all_important_sentences_pubmed[sentence]
            if pubmed not in metadata['pubmed_sentences_annotations'] or \
                    'pubmed_sentences_relation_annotations' not in \
                    metadata['pubmed_sentences_annotations'][pubmed]:
                continue
            relations = metadata['pubmed_sentences_annotations'][pubmed]\
                    ['pubmed_sentences_relation_annotations'][0]
            added_sentences = set()
            for r in relations:
                if r[0] in sentence and r[1] in sentence:
                    x = r[0]
                    y = r[1]
                    test_sentence = sentence.replace(x, ' <X> '+x+' </X> ')
                    test_sentence = test_sentence.replace(y, ' <Y> '\
                                                        +y+' </Y> ').strip()
                    test_sentence = ' '.join(test_sentence.split()).strip()
                    f_causes.write("0\t0\t"+test_sentence+"\n")
                    to_annotate_sentence_original[test_sentence] = sentence
                    relation_sentences_considered.add(sentence)
            if 'pubmed_sentences_entity_annotations' not in \
                    metadata['pubmed_sentences_annotations'][pubmed]:
                continue
            for pubmed_sentence, sentence_entities in \
                    metadata['pubmed_sentences_annotations'][pubmed]\
                    ['pubmed_sentences_entity_annotations']:
                if sentence != pubmed_sentence:
                    continue
                x_entities = _get_entities(pubmed_sentence.split(),\
                        sentence_entities.split(), 'Food') + _get_entities(\
                        pubmed_sentence.split(), sentence_entities.split(),\
                        'Nutrition')
                y_entities = _get_entities(pubmed_sentence.split(),\
                        sentence_entities.split(), 'Condition')
                for x in x_entities:
                    for y in y_entities:
                        test_sentence = pubmed_sentence.replace(x, \
                                ' <X> '+x+' </X> ')
                        test_sentence = test_sentence.replace(y, ' <Y> '\
                                +y+' </Y> ').strip()
                        test_sentence = ' '.join(test_sentence.split()).strip()
                        to_annotate_sentence_original[test_sentence] = \
                                pubmed_sentence
                        if test_sentence not in relation_sentences_considered:
                            f_causes.write("0\t0\t"+test_sentence+"\n")
                            relation_sentences_considered.add(test_sentence)
        f_causes.close()
        f_sentiment.close()


    all_important_pubmed_phrases = {}
    lines = open(split_name + "_importance_pubmed_general.txt"\
            ".rationale.machine_prob_readable.tsv","r").readlines()[1:]#\
            #+ open(split_name + "_importance_pubmed_sentiment.txt"\
            #".rationale.machine_prob_readable.tsv","r").readlines()[1:]

    pubmed_sentence_causal_rationales = {}
    pubmed_sentence_causal_rationales_importance = {}


    if args.run_features:
        predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/"\
                        "allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

    not_part_of_to_annotations = 0

    pubmed_sentence_verb_phrase_rationales = {}
    pubmed_sentence_verb_phrases = {}
    pubmed_sentence_constituency_parser = {}
    if os.path.exists("/tmp/pubmed_sentence_constituency_parse.p"):
        pubmed_sentence_constituency_parser = \
                pickle.load(open("/tmp/pubmed_sentence_constituency_parse.p","rb"))
        pubmed_sentence_verb_phrases = \
                pickle.load(open("pubmed_sentence_verb_phrases.p","rb"))

    parsing_sentences_considered = set()

    for line in tqdm(lines):
        line = line.strip()
        parts= line.split("\t")
        if float(parts[1]) == 0.0 and False:
            continue
        importance_prob = float(parts[1])
        selected_tokens = []
        for token, rationale in zip(parts[2].split(), parts[3].split()):
            if int(rationale) == 1 or True:
                selected_tokens.append(token)
        selected_tokens = ' '.join(selected_tokens).strip()
        span = selected_tokens
        #span = selected_tokens.replace('<X>','').replace('</X>','')\
        #        .replace('<Y>','').replace('</Y>','')

        if ' '.join(parts[2].strip().split()).strip() \
                not in to_annotate_sentence_original:
            not_part_of_to_annotations += 1
            continue

        x_string = span[span.find("<X>")+3:span.find("</X>")].strip()
        y_string = span[span.find("<Y>")+3:span.find("</Y>")].strip()

        #if tuple([x_string,y_string]) not in all_cause_tuples:
        #    not_part_of_to_annotations += 1
        #    continue

        original_sentence = \
                ' '.join(to_annotate_sentence_original[\
                ' '.join(parts[2].strip().split()).strip()].split())\
                .strip()
        parsing_sentences_considered.add(original_sentence)
        if original_sentence not in pubmed_sentence_constituency_parser\
                and args.run_features:
            pubmed_sentence_constituency_parser[original_sentence] = \
                    predictor.predict(sentence=original_sentence)
        if original_sentence not in pubmed_sentence_verb_phrases\
                and args.run_features:
            verb_phrases = []
            stack = []
            if 'children' in pubmed_sentence_constituency_parser\
                    [original_sentence]['hierplane_tree']['root']:
                stack += pubmed_sentence_constituency_parser[\
                        original_sentence]['hierplane_tree']\
                        ['root']['children']
            while len(stack) > 0:
                child = stack[0]
                if 'children' in child:
                    stack += child['children']
                if child['nodeType'] == 'VP':
                    verb_phrases.append(child['word'])
                del stack[0]
            pubmed_sentence_verb_phrases[original_sentence] = verb_phrases
        best_verb_phrase = original_sentence
        for verb_phrase in pubmed_sentence_verb_phrases[original_sentence]:
            if y_string not in verb_phrase:
                continue
            if len(verb_phrase) < len(best_verb_phrase):
                best_verb_phrase = verb_phrase
        pubmed_sentence_verb_phrase_rationales[original_sentence] = \
                pubmed_sentence_verb_phrase_rationales.setdefault(original_sentence,\
                []) + [best_verb_phrase]

        new_importance_prob = all_important_sentences_importance_probabilities\
                [original_sentence]
        if 'NCBI' in span:
            span = span.replace('NCBI','')
        if 'PubMed' in span:
            span = span.replace('PubMed','')
        if 'NCBI' in best_verb_phrase:
            best_verb_phrase = best_verb_phrase.replace('NCBI','')
        if 'PubMed' in best_verb_phrase:
            best_verb_phrase = best_verb_phrase.replace('PubMed','')
        pubmed_sentence_causal_rationales[original_sentence] = \
                pubmed_sentence_causal_rationales.setdefault(original_sentence,\
                []) + [span + " ### " + best_verb_phrase]\
                #[parts[2] + " ### " + span]#[span + " ### " + best_verb_phrase]
        pubmed_sentence_causal_rationales_importance[original_sentence] = \
                pubmed_sentence_causal_rationales_importance.setdefault(original_sentence,\
                []) + [new_importance_prob]#[importance_prob]

    print("%d not found" %not_part_of_to_annotations)

    print(len(relation_sentences_considered), len(parsing_sentences_considered)) 

    if args.run_features:
        pickle.dump(pubmed_sentence_constituency_parser,\
            open("/tmp/pubmed_sentence_constituency_parse.p","wb"))
        pickle.dump(pubmed_sentence_verb_phrases,\
            open("pubmed_sentence_verb_phrases.p","wb"))

    sentence_importance_file\
                    = jsonlines.open("T4_dev_general.jsonl","w")
    character_importance_file\
            = jsonlines.open("importance_dev.jsonl","w")
    input_dicts          = []
    for pubmed_sentence in pubmed_sentence_causal_rationales:
        for rationale in pubmed_sentence_causal_rationales[pubmed_sentence]:
            x_string = rationale[rationale.find("<X>")+3:\
                    rationale.find("</X>")].strip()
            y_string = rationale[rationale.find("<Y>")+3:\
                    rationale.find("</Y>")].strip()
            c_dict     = {'sentence':x_string + " # " + y_string,\
                        'rationale': rationale, 'pubmed_sentence':\
                        pubmed_sentence, 'gold_label':'increases'}
            character_importance_file.write(c_dict)
            dict = {'sentence':rationale, 'rationale': rationale,\
                    'pubmed_sentence': pubmed_sentence, 'gold_label':\
                    'increases'}
            input_dicts.append(dict)
            sentence_importance_file.write(dict)
    character_importance_file.close()
    sentence_importance_file.close()
    if args.run_features:
        os.chdir("/data/rsg/nlp/darsh/pytorch-pretrained-BERT")
        os.system("python examples/run_causes.py --task_name re_task "\
            "--do_eval --do_lower_case --data_dir "\
            "/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/"\
            " --bert_model bert-base-uncased --max_seq_length 128"\
            " --train_batch_size 32 --learning_rate 5e-5"\
            " --num_train_epochs 3.0 --output_dir "\
            "t4_general_causes_output --output_preds")
        os.system("python examples/run_importance.py --task_name re_task "\
                "--do_eval --do_lower_case --data_dir "\
                "/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/"\
                " --bert_model bert-base-uncased --max_seq_length 128"\
                " --train_batch_size 32 --learning_rate 5e-5"\
                " --num_train_epochs 3.0 --output_dir "\
                "importance_classification --output_preds")
        os.chdir("/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT")
        copyfile("/data/rsg/nlp/darsh/"\
            "pytorch-pretrained-BERT/t4_general_causes_output/preds.jsonl",\
            "sentence_importance_"+split_name+"_preds.jsonl")
        copyfile("/data/rsg/nlp/darsh/"\
                "pytorch-pretrained-BERT/importance_classification/preds.jsonl",\
                "character_importance_"+split_name+"_preds.jsonl")
    predictions = jsonlines.open("sentence_importance_"+split_name+\
            "_preds.jsonl","r")
    predictions2= jsonlines.open("character_importance_"+split_name+\
            "_preds.jsonl","r")
    for input_dict, prediction, prediction2 in zip(input_dicts, \
            predictions, predictions2):
        pubmed_sentence = input_dict['pubmed_sentence']
        rationale       = input_dict['rationale']
        x_string        = rationale[rationale.find("<X>")+3:\
                rationale.find("</X>")].strip()
        y_string        = rationale[rationale.find("<Y>")+3:\
                                rationale.find("</Y>")].strip()
        #assert tuple([x_string, y_string]) in all_cause_tuples
        importance_prob = float(prediction['increases'])
        character_prob  = float(prediction2['increases'])
        rationale_index = pubmed_sentence_causal_rationales[pubmed_sentence]\
                .index(rationale)
        pubmed_sentence_causal_rationales_importance[pubmed_sentence]\
                [rationale_index] = [character_prob,\
                importance_prob]#character_prob#importance_prob * character_prob


    print("Found %d sentences from the pubmed articles." %number_of_sentences)

    sentence_pubmed_sentences = {}
    sentence_pubmed_pubmeds   = {}
    sentence_causes = {}
    sentence_contains={}
    main_keep_sentences = set()

    darsh_pubmed_causes = pickle.load(open("pubmed_causes.p","rb"))

    extractive_output_summaries  = jsonlines.open(\
            "extractive_multi_summaries.jsonl","w")
    extractive_permuted_output_summaries = \
            jsonlines.open("extractive_perm_multi_summaries.jsonl","w")

    epoch_rewards = []
    epoch_missed  = []

    split_sentences = set()
    sentences_with_causes = set()

    total_epochs = args.epochs

    sentence_keys= list(sentence_all_causes.keys())

    all_cause_templates = []
    for x in sentence_extracted_rationales.values():
        if 'causes' in x:
            if len(x['causes']) != 1 or \
                x['causes'][0][x['causes'][0].find("<X>"):\
                x['causes'][0].find("</X>")].strip() == "" or\
                x['causes'][0][x['causes'][0].find("<Y>"):\
                x['causes'][0].find("</Y>")].strip() == "" or\
                x['causes'][0].count('<X>') != 1 or \
                x['causes'][0].count('</X>') != 1 or \
                x['causes'][0].count('<Y>') != 1 or \
                x['causes'][0].count('</Y>') != 1:
                    continue
            all_cause_templates += x['causes']


    f = open("missed.txt","w")

    current_input_causes = {}
    inferred_causes = {}
    cached_inputs   = {}

    for I in tqdm(range(total_epochs)):
        current_considered = 0
        average_sentences  = []
        avg_max_candidates = []
        if split_name != 'train' or I == total_epochs-1:
            policy.eval()
        gold_sentence_consider_sentences = {}
        rewards = []
        bad_sentences = 0
        correctly_bad = 0
        missed_both_cases = 0
        policy_misses = 0
        prev_loss = []
        gold_causes_count = []
        min_templates_required = {}
        new_min_templates_required = {}
        number_successful = {}

        if args.data_points == -1 and args.split_name == 'train':
            random.shuffle(sentence_keys)
        else:
            sorted(sentence_keys)
        

        for darsh in range(1):
            for sentence in tqdm(sentence_keys):
                title_food_name = ""
                if len(title_entities.get(sentence_file_names[sentence],[""]))>0:
                    title_food_name = title_entities.get(sentence_file_names[sentence],[""])[0]
                dict = {'file_name':sentence_file_names[sentence], 'gold':sentence,\
                        'output':'', 'outputs':[], 'consider_tuples':None, \
                        'food_name':title_food_name}
                if len(average_sentences) == args.data_points:
                    break
                if metadata[sentence_file_names[sentence]]['split'] != split_name:
                    continue
                split_sentences.add(sentence)
                gold_sentence_consider_sentences[sentence] = dict
                sentence_causes[sentence] = []
                sentence_pubmed_sentences[sentence] = {}
                sentence_pubmed_pubmeds[sentence]   = {}
                for sent,pubmeds in metadata[sentence_file_names[sentence]]\
                        ['summary_inputs']['summary_pubmed_articles'].items():
                    if sent.strip() != sentence:
                        continue
                    for pubmed in pubmeds:
                        if pubmed not in pubmed_important_sentences:
                            continue
                        for sentiment in pubmed_important_sentences[pubmed]:
                            if pubmed_important_sentences[pubmed][sentiment] in \
                                sentence_pubmed_sentences[sentence].get(sentiment,[]):
                                continue
                            sentence_pubmed_sentences[sentence][sentiment] = \
                                sentence_pubmed_sentences[sentence].setdefault(\
                                sentiment,[]) + pubmed_important_sentences[pubmed]\
                                [sentiment]
                            sentence_pubmed_pubmeds[sentence][sentiment] = \
                                sentence_pubmed_pubmeds[sentence].setdefault(\
                                sentiment,[]) + [pubmed] * \
                                len(pubmed_important_sentences[pubmed])
                for sentiment in ['Bad','Good','Neutral'][:1]:
                    consider_sentences = []
                    for i in range(1):
                        if len(sentence_pubmed_sentences[sentence].get('Bad',[]))+\
                                 len(\
                                sentence_pubmed_sentences[sentence].get(\
                                'Good',[])) +\
                                len(sentence_pubmed_sentences[sentence].get(\
                                'Neutral',[])) >= 1:
                            batch_sentences = []
                            batch_importance= []
                            batch_sentiment = []
                            batch_pubmed    = []
                            for sent,pubmed in\
                                    zip(sentence_pubmed_sentences[sentence]\
                                    .get('Good',[]),\
                                    sentence_pubmed_pubmeds[sentence].get('Good',[])):
                                batch_sentences += \
                                    pubmed_sentence_causal_rationales.get(sent,[])
                                batch_importance += \
                                    pubmed_sentence_causal_rationales_importance\
                                    .get(sent,[])
                                batch_sentiment += [[1,0,0]]*len(\
                                    pubmed_sentence_causal_rationales_importance.get(\
                                    sent,[]))
                                batch_pubmed += [pubmed]*len(\
                                        pubmed_sentence_causal_rationales_importance.get(\
                                        sent,[]))

                            for sent,pubmed in zip(\
                                    sentence_pubmed_sentences[sentence].get(\
                                    'Bad',[]),\
                                    sentence_pubmed_pubmeds[sentence].get('Bad',[])):
                                batch_sentences += \
                                    pubmed_sentence_causal_rationales.get(sent,[])
                                batch_importance += \
                                    pubmed_sentence_causal_rationales_importance\
                                    .get(sent,[])
                                batch_sentiment += [[0,1,0]]*len(\
                                    pubmed_sentence_causal_rationales_importance.get(\
                                    sent,[]))
                                batch_pubmed +=  [pubmed]*len(\
                                        pubmed_sentence_causal_rationales_importance.get(\
                                        sent,[]))
                            for sent,pubmed in zip(sentence_pubmed_sentences[sentence].get(\
                                    'Neutral',[]),\
                                    sentence_pubmed_pubmeds[sentence].get('Neutral',[])):
                                batch_sentences += \
                                    pubmed_sentence_causal_rationales.get(sent,[])
                                batch_importance += \
                                    [x for x in pubmed_sentence_causal_rationales_importance\
                                    .get(sent,[])]
                                batch_sentiment += [[0,0,1]]*len(\
                                        pubmed_sentence_causal_rationales_importance.get(\
                                    sent,[]))
                                batch_pubmed    += [pubmed]*len(\
                                        pubmed_sentence_causal_rationales_importance.get(\
                                        sent,[]))

                            b_sentences = []
                            b_importance= []
                            b_sentiment = []
                            b_type      = []
                            b_pubmed    = []
                            current_input_causes[sentence] = set()
                            assert len(batch_sentences) == len(batch_pubmed)
                            for b_s, b_i, b_senti, b_pub in zip(batch_sentences,\
                                    batch_importance, batch_sentiment, batch_pubmed):
                                if b_s is not None and b_s not in b_sentences:
                                    if "<X>" in b_s and "</X>" in b_s and "<Y>"\
                                           in b_s and "</Y>" in b_s:
                                        x_string = b_s[b_s.find("<X>")+3:\
                                            b_s.find("</X>")].strip()
                                        y_string = b_s[b_s.find("<Y>")+3:\
                                            b_s.find("</Y>")].strip()
                                        if tuple([x_string,y_string]) not in \
                                                all_cause_sentence_tuples[sentence]:
                                            f.write(b_s + "\n")
                                            continue
                                    b_sentences.append(b_s)
                                    b_importance.append(b_i)
                                    b_sentiment.append(b_senti)
                                    b_pubmed.append(b_pub)
                                    if "<X>" in b_s and "</X>" in b_s and "<Y>"\
                                            in b_s and "</Y>" in b_s:
                                        b_type.append(1)
                                        current_input_causes[sentence].add(tuple([x_string,y_string]))
                                    else:
                                        continue
                                        b_type.append(0)
                            batch_sentences = b_sentences
                            batch_importance= b_importance
                            batch_sentiment = b_sentiment
                            batch_type      = b_type
                            batch_pubmed    = b_pubmed
                            assert len(batch_sentiment) == len(batch_sentences)
                            assert not any([x is None for x in batch_sentences])
                            if len(batch_sentences) == 0:
                                if len(sentence_all_causes[sentence]) > 0:
                                    possible_things = set()
                                    pubmed_candidates = metadata[sentence_file_names[sentence]]\
                                            ["pubmed_sentences"].keys()
                                    for pubmed_candidate in pubmed_candidates:
                                        if pubmed_candidate not in metadata['pubmed_sentences_annotations']:
                                            continue
                                        if 'pubmed_sentences_relation_annotations' not in \
                                        metadata['pubmed_sentences_annotations'][pubmed_candidate]\
                                            or 'pubmed_sentences_entity_annotations' not in \
                                        metadata['pubmed_sentences_annotations'][pubmed_candidate]:
                                            continue
                                        relations = metadata\
                                                ['pubmed_sentences_annotations']\
                                                [pubmed_candidate]\
                                                ['pubmed_sentences_relation_annotations'][0]
                                        for relation in relations:
                                            for sentence_p,labels in metadata['pubmed_sentences_annotations']\
                                                    [pubmed_candidate]['pubmed_sentences_entity_annotations']:
                                                        if relation[0] in sentence and relation[1] in sentence:
                                                            possible_things.add(tuple(relation + \
                                                                    [sentence_p]))
                                break
                            assert len(batch_sentences) > 0
                            if len(batch_sentences) == 1:
                                consider_sentences = batch_sentences
                            else:
                                not_1 = False
                                batch_structures = []
                                for b_s,b_t,b_p in zip(batch_sentences,batch_type,\
                                        batch_pubmed):
                                    if b_t == 1:
                                        x_string = b_s[b_s.find("<X>")+3:\
                                                b_s.find("</X>")].strip()
                                        y_string = b_s[b_s.find("<Y>")+3:\
                                                b_s.find("</Y>")].strip()
                                        batch_structures.append([x_string,\
                                                 y_string, b_s, b_p])
                                causes_count = [_compare_causes(g_r, batch_structures,\
                                        embeddings, "") for g_r in \
                                        gold_sentence_causes[sentence]]
                                causes_2d    = [[_compare_causes(g_r, [structure],\
                                        embeddings, "") for g_r in \
                                        gold_sentence_causes[sentence]] \
                                                for structure in \
                                        batch_structures]
                                if any(causes_count) and len(causes_count) > 0:
                                    if any([sum(x) == sum(causes_count)\
                                            for x in causes_2d]):
                                        min_templates_required[1] = \
                                                min_templates_required.\
                                                setdefault(1, 0) + 1
                                    else:
                                        min_templates_required["not 1"] = \
                                                min_templates_required.\
                                                setdefault("not 1", 0) + 1
                                        not_1 = True
    
                                if args.split_name == 'train' and \
                                        sum(causes_count) > 1:
                                    ignore_indices = set()
                                    for cause_ind in range(len(causes_2d)):
                                        if causes_2d[cause_ind].count(1.0) == \
                                                len(causes_2d[cause_ind]):
                                            pass
                                    n_batch_structures = []
                                    n_batch_importance = []
                                    n_batch_sentiment  = []
                                    n_batch_type       = []
                                    n_causes_2d        = []
                                    for cause_ind in range(len(batch_structures)):
                                        if cause_ind in ignore_indices and \
                                                len(ignore_indices) < len(batch_structures)-2:
                                            continue
                                        n_batch_structures.append(batch_structures[cause_ind])
                                        n_batch_importance.append(batch_importance[cause_ind])
                                        n_batch_sentiment.append(batch_sentiment[cause_ind])
                                        n_batch_type.append(batch_type[cause_ind])
                                        n_causes_2d.append(causes_2d[cause_ind])
                                    batch_structures = n_batch_structures
                                    batch_importance = n_batch_importance
                                    batch_sentiment  = n_batch_sentiment
                                    batch_type       = n_batch_type
                                    causes_2d        = n_causes_2d

                                if any(causes_count) and len(causes_count) > 0:
                                    if any([sum(x) == sum(causes_count)\
                                        for x in causes_2d]):
                                        new_min_templates_required['1'] = \
                                        new_min_templates_required.setdefault('1',0)+1
                                    else:
                                        new_min_templates_required['not 1'] = \
                                        new_min_templates_required.setdefault('not 1',\
                                    0)+1

                                if len(gold_sentence_causes[dict['gold']]) > 0 and\
                                    any(causes_count) and args.task=='policy':
                                    gold_causes_count.append(len(gold_sentence_causes[dict['gold']]))
                                    pubmed_sentences = {}
                                    for pubmed,p_sentence_tuples in metadata[dict['file_name']]\
                                            ['pubmed_sentences'].items():
                                        text = ""
                                        p_ind = 0
                                        for p_s,p_t in p_sentence_tuples:
                                            text += " ".join(p_s).strip() + " "
                                            if p_ind == 0:
                                                text+="### "
                                            p_ind += 1
                                        pubmed_sentences[pubmed] = text.strip()
                                    env = Environment(gold_sentence_causes[\
                                        sentence], \
                                        sentence_population_entities\
                                        [sentence][0],embeddings,\
                                        np.concatenate(\
                                            ((property_style_rationales_representations['causes'],\
                                        property_style_rationales_representations['contains'],\
                                        property_style_rationales_representations['sentiment'])\
                                        ),axis=0),
                                    sentence_extracted_rationale_representations.get(sentence,\
                                                np.zeros((1,50))),
                                    all_gold_sentence_extracted_representations,\
                                    all_pubmed_rationale_representations, rhs,\
                                    dict['gold'],pubmed_sentences=pubmed_sentences)
                                    batch_structures = []
                                    n_batch_importance = []
                                    n_batch_type = []
                                    n_batch_sentiment = []
                                    structures_sentences= {}
                                    for b_s,b_t,b_p,b_i,b_sent in zip(batch_sentences,\
                                            batch_type,batch_pubmed,batch_importance,\
                                            batch_sentiment):
                                        pubmed_population = []
                                        if b_p in \
                                                metadata['pubmed_sentences_annotations']:
                                            entity_annotations = \
                                                metadata['pubmed_sentences_annotations']\
                                                [b_p]['pubmed_sentences_entity_annotations']
                                            all_populations = []
                                            for element in entity_annotations:
                                                element_population = _get_entities(\
                                                        element[0].split(),\
                                                        element[1].split(),\
                                                        'Population')
                                                if element[0] == b_s and len(element_population)>0:
                                                    pubmed_population = element_population
                                                    break
                                                else:
                                                    pubmed_population += element_population
                                        pubmed_population = list(set(pubmed_population))
                                        pubmed_population = " ".join(pubmed_population).strip() 
                                        if b_t == 1:
                                            x_string = b_s[b_s.find("<X>")+3:\
                                                    b_s.find("</X>")].strip()
                                            y_string = b_s[b_s.find("<Y>")+3:\
                                                    b_s.find("</Y>")].strip()
                                            batch_structures.append(tuple([x_string,\
                                                    y_string, b_s, b_p, \
                                                    pubmed_population]))
                                            structures_sentences[tuple([x_string,\
                                                y_string, b_s])] = b_s
                                            n_batch_importance.append(b_i)
                                            n_batch_sentiment.append(b_sent)
                                            n_batch_type.append(b_t)
                                        else:
                                            batch_structures.append(tuple(b_s))
                                            structures_sentences[tuple(b_s)] = b_s
                                    batch_importance = n_batch_importance
                                    batch_type       = n_batch_type
                                    batch_sentiment  = n_batch_sentiment
                                    assert len(batch_structures) > 0
                                    if True:
                                        if split_name != 'train' or \
                                                I == total_epochs-1:
                                            policy.eval()
                                        batch_extractions = [\
                                                x[2][x[2].find("###")+3:].strip()\
                                                .replace('NCBI', '').\
                                                replace('PubMed','')for x in \
                                                batch_structures]
                                        batch_representations = \
                                                [_get_phrase_embeddings(extraction\
                                                , embeddings) for extraction in \
                                                batch_extractions]
                                        assert len(batch_structures) == \
                                                len(batch_representations)
                                        for ind,batch_rep in enumerate(\
                                                batch_representations):
                                            batch_structures[ind] =\
                                                    tuple(list(batch_structures[ind][:3]) + \
                                                    [batch_rep] + \
                                                    list(batch_structures[ind][3:]))
                                        main_keep_sentences.add(sentence)
                                        d_batch_structures = []
                                        d_batch_importance = []
                                        d_batch_sentiment  = []
                                        d_batch_type       = []
                                        d_batch_representations = [] 
                                        darsh_pubmed_considered = metadata[dict['file_name']]\
                                                ['summary_inputs']\
                                                ['summary_pubmed_articles']
                                        darsh_pubmed_considered = {x.strip():y for x,y in \
                                                darsh_pubmed_considered.items()}
                                        pubmed_considered = darsh_pubmed_considered[dict['gold']]
                                        for p_c in pubmed_considered:
                                            c_c = darsh_pubmed_causes.get(p_c,[])
                                            for d_b in c_c:
                                                if d_b[2] == 'None':
                                                    continue
                                                d_batch_structure = [d_b[0],d_b[1],\
                                                        d_b[2]+" <X>"+\
                                                        d_b[0]+"</X> <Y>"+d_b[1]+"</Y> ",\
                                                        np.array([0]*50),p_c,'']
                                                d_batch_structures.append(tuple(\
                                                        d_batch_structure))
                                                d_batch_importance.append(1.0)
                                                d_batch_sentiment.append([1,0,0])
                                                d_batch_type.append(1)
                                                d_batch_representations.append(np.array([0]*50))
                                        batch_structures = d_batch_structures
                                        batch_importance = d_batch_importance
                                        batch_sentiment  = d_batch_sentiment
                                        batch_type       = d_batch_type
                                        batch_representations = d_batch_representations
                                        if len(batch_structures) == 0:
                                            continue
                                        consider_tuples, reward, prev_loss =\
                                                _get_choice_policy_sentences(env,\
                                                batch_structures,\
                                                batch_importance,\
                                                batch_sentiment,\
                                                batch_type,\
                                                batch_representations,\
                                                policy, optimizer,\
                                                gold_sentence_sentiments[sentence],\
                                                vectorizer,\
                                                property_style_rationales_representations['causes'],\
                                                prev_loss, batch_size=args.batch_size,\
                                                epoch_number=I, pretrain_until=0,\
                                                repeat_instance=\
                                                args.repeat_instance if not_1\
                                                else args.repeat_instance)
                                        dict['consider_tuples'] = \
                                                consider_tuples
                                    inferred_causes[sentence] = consider_tuples
                                    rewards.append(reward)
                                    current_considered += 1
                                    average_sentences.append(len(consider_tuples))
                                    avg_max_candidates.append\
                                            (min(5,len(batch_structures)))
                                    consider_sentences = []
                                    for consider_tuple in consider_tuples:
                                        if tuple(consider_tuple[:3]) in \
                                            structures_sentences:
                                            consider_sentences.append(\
                                                        structures_sentences\
                                                        [tuple(consider_tuple[:3])])
                                            break
                                    if len(consider_sentences) == 0:
                                        missed_both_cases += 1
                                        policy_misses += 1
                                else:
                                    consider_sentences = batch_sentences
                                    consider_sentences = _get_clustered_sentences(\
                                    batch_sentences, batch_importance, vectorizer, \
                                    (2-i/10)*cluster_threshold)
                                    for consider_sentence in consider_sentences:
                                        consider_sentence = consider_sentence.lower()
                                        x_string = consider_sentence[consider_sentence.find("<x>")+3:\
                                                consider_sentence.find("</x>")].strip()
                                        y_string = consider_sentence[consider_sentence.find("<y>")+3:\
                                                consider_sentence.find("</y>")].strip()
                                        inferred_causes[sentence] = \
                                                inferred_causes.get(sentence,[]) + \
                                                [[x_string,y_string]]
                        else:
                            consider_sentences = sentence_pubmed_sentences[sentence]\
                                .get('Good',[]) + \
                                sentence_pubmed_sentences[sentence].get('Bad',[]) + \
                                sentence_pubmed_sentences[sentence].get('Neutral',[])
                        candidates = []
                        for cause_triplet in sentence_all_causes[sentence]:
                            if cause_triplet[1].lower() in \
                            ' '.join(consider_sentences).strip():
                                if any([" <x> " + cause_triplet[0].lower() + " </x> "\
                                        in x.lower() and \
                                    " <y> " + cause_triplet[1].lower() + " </y> " in x.lower()\
                                    for x in consider_sentences]):
                                        sentence_causes[sentence] = sentence_causes.setdefault(\
                                    sentence,[]) + [cause_triplet]
                        if len(sentence_causes[sentence]) == 0:
                            sentence_causes[sentence] = candidates
                        if len(sentence_causes[sentence]) > 1:
                            sentence_pubmed_sentences[sentence] = consider_sentences
                            if sentiment == 'Bad':
                                bad_sentences += 1
                                if sentence_sentiment[sentence] == sentiment:
                                    correctly_bad += 1
                            break
                    if len(consider_sentences) > 0:
                        dict['outputs'] = list(consider_sentences)
                        dict['output']  = ' '.join(\
                                list(consider_sentences)).strip()
                        assert sentence in split_sentences
                        gold_sentence_consider_sentences[sentence] = dict
                    if len(sentence_causes.get(sentence,[])) > 1:
                        dict['outputs'] = list(consider_sentences)
                        dict['output']  = ' '.join(\
                                list(consider_sentences)).strip()
                        assert sentence in split_sentences
                        gold_sentence_consider_sentences[sentence] = dict
                        break
                    consider_sentences_set = set()
                    new_consider_sentences = []
                    for consider_sentence in consider_sentences:
                        consider_clean_sentence = re.sub('<[^>]+>', '', \
                                consider_sentence)
                        consider_clean_sentence = consider_clean_sentence.replace(\
                                '<','').replace('>','')
                        consider_clean_sentence = re.sub(' +', ' ',\
                                consider_clean_sentence)
                        if consider_clean_sentence not in consider_sentences_set:
                            new_consider_sentences.append(consider_sentence)
                    consider_sentences = new_consider_sentences
                    dict['outputs'] = list(consider_sentences)
                    dict['output']  = ' '.join(\
                        list(consider_sentences)).strip()
                    assert sentence in split_sentences  
                    gold_sentence_consider_sentences[sentence] = dict
            epoch_rewards.append(sum(rewards)/len(rewards) if len(rewards)>0 else 0)
        print("Epoch Rewards ", epoch_rewards)
        epoch_missed.append(policy_misses)
        print(min_templates_required)
        print(new_min_templates_required)
        if args.task == 'policy':
            print(policy_misses, len(average_sentences),  \
                sum(average_sentences)/len(average_sentences),\
                sum(avg_max_candidates)/len(avg_max_candidates),\
                sum(gold_causes_count)/len(gold_causes_count))
        if split_name != 'train':
            break
        if split_name == 'train':
            torch.save(policy.state_dict(),open('choice_policy_%d.pt' %(I+1),'wb'))
    
    print(len(split_sentences), len(gold_sentence_consider_sentences))
    torch.save(policy.state_dict(),open('choice_policy.pt','wb'))

    f.close()

    plt.plot(epoch_rewards[:-1])
    plt.ylabel("Rewards")
    plt.savefig('rewards.png')
    plt.plot(epoch_missed[:-1])
    plt.ylabel("Missed Cases")
    plt.savefig("missed.png")

    matching_counts = 0
    missed_counts   = 0

    print("Trained on %d main sentences" %len(main_keep_sentences))

    for sentence in sentence_all_contains:
        if sentence not in gold_sentence_consider_sentences:
            continue
        if metadata[sentence_file_names[sentence]]['split'] != split_name:
            continue
        if sentence == "Raw legumes harbor antinutrients, which may cause harm."\
                " However, proper preparation methods get rid of most of them.":
            pass
        if sentence in gold_sentence_consider_sentences:
            dict = gold_sentence_consider_sentences[sentence]
            consider_sentences = dict['outputs']
            predicted_populations = None if dict['consider_tuples'] is None else\
                    [x[5] for x in dict['consider_tuples']]
            gold_populations      = sentence_population_entities[\
                    dict['gold']][0]
            if predicted_populations is None or \
                    all([len(x)==0 for x in predicted_populations]):
                if len(gold_populations) == 0:
                    matching_counts += 1
                else:
                    missed_counts += 1
            else:
                if len(gold_populations) > 0:
                    matching_counts += 1
                else:
                    missed_counts += 1
            sentence_specific_contains  = {}
            for consider_sentence in consider_sentences:
                x_string = consider_sentence[consider_sentence.find("<X>")+3:\
                        consider_sentence.find("</X>")].strip()
                candidate_contains = []
                for contain in sentence_all_contains.get(sentence,[]):
                    if matching_score(contain[1].lower(),\
                            x_string.lower(),embeddings) > 0.85:
                        candidate_contains.append(contain)
                best_candidate_contains = None
                best_cosine_score       = -1.0
                for candidate_contain in candidate_contains:
                    if \
                        len(title_entities.get(sentence_file_names[sentence],\
                        [''])) == 0:
                        continue
                    current_cosine = matching_score(title_entities.get(\
                            sentence_file_names[sentence],[''])\
                            [0].lower(),\
                            candidate_contain[0].lower(),embeddings)
                    if current_cosine > best_cosine_score and current_cosine > 0.85:
                        best_cosine_score = current_cosine
                        best_candidate_contains = candidate_contain
                if best_candidate_contains is None and\
                        len(title_entities.get(\
                        sentence_file_names[sentence],[])) > 0\
                        and matching_score(title_entities.get(\
                        sentence_file_names[sentence],[''])\
                        [0].lower(),\
                        x_string, embeddings) < 0.85:
                    best_candidate_contains = [title_entities.get(\
                            sentence_file_names[sentence],[''])[0],\
                             x_string]
                sentence_specific_contains[consider_sentence] = best_candidate_contains

            new_consider_sentences = []
            new_tupled_consider_sentences = []
            for consider_sentence in \
                        consider_sentences:
                new_consider_sentences.append(consider_sentence)
                new_tupled_consider_sentences.append([consider_sentence])
            gold_sentence_consider_sentences[sentence]['outputs'] = \
                    new_consider_sentences
            gold_sentence_consider_sentences[sentence]['perm_outputs'] = \
                    new_tupled_consider_sentences
            gold_sentence_consider_sentences[sentence]['output'] = \
                    (' '.join(new_consider_sentences)).strip()

        sentence_contains[sentence] = []
        already_computed = False
        for sent,pubmeds in metadata[sentence_file_names[sentence]]\
                ['summary_inputs']['summary_pubmed_articles'].items():
            if sent.strip() != sentence:
                continue
            if sentence in sentence_pubmed_sentences:
                already_computed = True
                break
            for pubmed in pubmeds:
                sentence_pubmed_sentences[sentence] = \
                        sentence_pubmed_sentences.setdefault(sentence,[]) + \
                        pubmed_important_sentences.get(pubmed, []) 

        for sentiment in ['Bad','Good']:
            if sentiment in sentence_pubmed_sentences[sentence]\
                and not already_computed:
                sentence_pubmed_sentences[sentence] = _get_clustered_sentences(\
                sentence_pubmed_sentences[sentence][sentiment], \
                [1.0] * sentence_pubmed_sentences[sentence][sentiment], \
                vectorizer, cluster_threshold)

            for contains_tuple in sentence_all_contains[sentence]:
                if contains_tuple[1].lower() in \
                        ' '.join(sentence_pubmed_sentences[sentence]).strip():
                    sentence_contains[sentence] = sentence_contains.setdefault(\
                        sentence,[]) + [contains_tuple]
            if len(sentence_contains[sentence]) > 0:
                break


    print("Matching population counts %d , missed %d" \
            %(matching_counts,missed_counts))

    print("Missed %d sentences for not having any importance stuff "\
            %missed_both_cases)
    print("Selected %d of actual as negative sentiment sentences" %correctly_bad)
    print("Selected %d of all as negative sentiment sentences." %bad_sentences)

    all_cause_templates_representations = vectorizer.transform(\
        all_cause_templates).todense()
    all_causes_tuples = []
    for cause_template in all_cause_templates:
        x_string = cause_template[cause_template.find("<X>"):\
                cause_template.find("</X>")].strip()
        y_string = cause_template[cause_template.find("<Y>"):\
                cause_template.find("</Y>")].strip()
        all_causes_tuples.append([x_string,y_string])

    templated_summaries_jsonl = jsonlines.open(\
                "templated_extractive_summaries.jsonl", "w")
    handwritten_summaries_jsonl = jsonlines.open(\
                "handwritten_summaries.jsonl","w")

    recalls = []
    all_input_recalls = []
    oracle_recalls = []
    s_ind = 0
    for sentence,dicti in tqdm(gold_sentence_consider_sentences.items()):
        s_ind += 1
        gold_causes = gold_sentence_causes.get(sentence,[])
        oracle_causes = sentence_all_causes.get(sentence,[])
        input_causes  = current_input_causes.get(sentence,[])
        infered_structures = inferred_causes.get(sentence,[])
        for output in dicti['outputs']:
            x_string = output[output.find("<X>")+3:\
                    output.find("</X>")].strip()
            y_string = output[output.find("<Y>")+3:\
                    output.find("<Y>")].strip()
            if x_string == "" or y_string == ""\
                    or "<X>" in x_string or "<Y>" in x_string\
                    or "<Y>" in y_string or "<Y>" in y_string:
                continue
            if '<X>' not in output or '<Y>' not in output:
                if 'contains' in output:
                    infered_structures.append(output.split('contains'))
                else:
                    pass
            else:
                infered_structures.append([x_string,y_string])
        old_consider_tuples = dicti['consider_tuples']
        if old_consider_tuples is None:
            dicti['consider_tuples'] = []
            for output in dicti['outputs']:
                x_string = output[output.find("<X>")+3:\
                        output.find("</X>")].strip()
                y_string = output[output.find("<Y>")+3:\
                        output.find("</Y>")].strip()
                dicti['consider_tuples'].append(tuple([x_string,y_string,\
                        output]))
        old_consider_tuples = dicti['consider_tuples']
        o_consider_tuples = []
        for consider_tuple in old_consider_tuples:
            if pubmed_entity_types.get(consider_tuple[0],\
                    []).count('Nutrition')>\
                    pubmed_entity_types.get(consider_tuple[0],\
                    []).count('Food') and len(title_entities.get(\
                    sentence_file_names.get(dicti['gold'],[])\
                    ,[]))\
                    >0:
                o_consider_tuples.append([title_entities[\
                        sentence_file_names[dicti['gold']]][0],\
                        consider_tuple[0]])
        old_consider_tuples += o_consider_tuples
        dicti['consider_tuples'] = old_consider_tuples
        correct_cases = 0
        oracle_cases  = 0
        input_cases   = 0
        #for gold_cause in gold_causes:
        #    correct_cases += _compare_causes(gold_cause, infered_structures,\
        #            embeddings,"")
        #    oracle_cases += _compare_causes(gold_cause, oracle_causes, \
        #            embeddings, "")
        #    input_cases  += _compare_causes(gold_cause, input_causes,\
        #            embeddings, "")
        if len(gold_causes) > 0:
            recalls.append(correct_cases/len(gold_causes))
            oracle_recalls.append(oracle_cases/len(gold_causes))
            all_input_recalls.append(input_cases/len(gold_causes))
        output_sentences = dicti['outputs']
        extractive_dicti = copy.copy(dicti)
        rationale_sentence = ""
        rationale_sentence_candidates = []
        for INd,p_output_sentences in enumerate(\
                [[[x] for x in dicti['outputs']]]):
            p_output_sentences = sum(p_output_sentences,[])
            rationale_sentence = ""
            added_nutrients    = set()
            for ind,sent in enumerate(p_output_sentences):
                if "###" in sent:
                    rationale = sent[sent.find("###")+3:].strip()
                    x_string  = sent[sent.find("<X>")+3:sent.find("</X>")]\
                        .strip()
                    if x_string.lower() not in rationale.lower() and \
                        (ind == 0 or x_string not in output_sentences[ind-1]):
                        rationale = x_string + " _ " + rationale + " . "
                else:
                    parts = [sent]
                    entity_type  = None
                    if ' contains ' in sent:
                        parts = sent.split(" contains ")
                        entity_type = 'Nutrition'
                    elif ' a ' in sent:
                        parts = sent.split(" a ")
                        entity_type = 'Food'
                    if len(parts) > 1 and np.dot(_get_phrase_embeddings(parts[0].lower(),\
                            embeddings),_get_phrase_embeddings(parts[1].lower(),embeddings\
                            )) < 0.85 and parts[1] not in added_nutrients:
                        nutrient = parts[1]
                        added_nutrients.add(nutrient)
                        if "###" in output_sentences[ind+1] and \
                                output_sentences[ind+1].split(\
                                "###")[1].strip().lower()\
                                .startswith(nutrient.lower()):
                            if entity_type == 'Nutrition':
                                rationale = parts[0] + " | "
                            else:
                                rationale = parts[0] + " * "
                        else:
                            if entity_type == "Nutrition":
                                rationale = parts[0] + " | " + parts[1]
                            else:
                                rationale = parts[0] + " * " + parts[1]
                    else:
                        rationale = parts[0]
                if rationale_sentence is not "":
                    if rationale_sentence.strip().endswith("|"):
                        rationale_sentence += rationale
                    else:
                        rationale_sentence += " _ " + rationale
                else:
                    rationale_sentence  = rationale
            extractive_dicti['rationale_output'] = rationale_sentence
            extractive_dicti['consider_tuples'] = \
                    [x[:2]+\
                    x[4:] if len(x)>3 else x for x in \
                    extractive_dicti['consider_tuples']]
            extractive_permuted_output_summaries.write(extractive_dicti)
            rationale_sentence_candidates.append(rationale_sentence) 
        extractive_dicti['rationale_output'] = rationale_sentence
        extractive_dicti['rationale_output_candidates'] = \
                rationale_sentence_candidates
        extractive_output_summaries.write(extractive_dicti)


    extractive_output_summaries.close()

    if args.split_name == 'train':
        time_stamp = str(time.time())
        torch.save(policy.state_dict(),open('choice_policy.pt_'+time_stamp,'wb'))

    print("Recall", sum(recalls)/len(recalls), \
            sum(oracle_recalls)/len(oracle_recalls),\
            sum(all_input_recalls)/len(all_input_recalls), len(recalls))

    return sentence_causes, sentence_contains




def get_templated_sentences(sentence_relation_annotations, \
        sentence_entity_annotations, sentence_causes, sentence_contains,\
        title_entities, sentence_file_names, metadata):

    output_file = jsonlines.open("templated_summaries.jsonl","w")
    healthline_graphs = []
    healthline_sentences = []
    for sentence in sentence_relation_annotations:
        output, _ = sentence_relation_annotations[sentence]
        data_points = output[0]
        for data_point in output[4]:
            data_points.append(data_point + ['contains'])
        healthline_graphs.append(get_causes_dict(data_points,\
                sentence_entity_annotations[sentence][0]))
        healthline_sentences.append(sentence)
    healthline_graphs = np.array(healthline_graphs)
    modified_sentences = {}
    template_scores = []
    for sentence in tqdm(sentence_causes):
        assert sentence in sentence_causes
        assert sentence in sentence_relation_annotations
        assert sentence in sentence_entity_annotations
        entity_types = {}
        if metadata[sentence_file_names[sentence]]['split'] != 'test':
            continue
        for entity in title_entities[sentence_file_names[sentence]]:
            entity_types[entity] = 'Food'
        data_points = sentence_causes[sentence]
        for data_point in sentence_contains[sentence]:
            data_points.append(data_point + ['contains'])
        causes_graph = get_causes_dict(data_points,\
                {**sentence_entity_annotations[sentence][1],\
                **entity_types})
        matching_scores = [match_dicts(causes_graph,ref_dict) \
                for ref_dict in healthline_graphs]
        matched_sentence = \
                healthline_sentences[matching_scores.index(max(matching_scores))]
        template_scores.append(max(matching_scores))
        if template_scores[-1] == 0.0:
            continue
        modified_sentences[sentence] = get_modified_sentence(\
                sentence_relation_annotations,\
                sentence_entity_annotations,\
                matched_sentence, data_points,\
                {**sentence_entity_annotations[sentence][1],\
                    **entity_types}, sentence)
        dict = {}
        dict['gold'] = sentence
        dict['output']= modified_sentences[sentence]
        dict['output_dictioanary'] = data_points
        dict['retrieved_dict'] = healthline_graphs[\
                matching_scores.index(max(matching_scores))]
        output_file.write(dict)
    output_file.close()
    print(sum(template_scores)/len(template_scores))
    return modified_sentences


def get_rationale_entities(file_name):

    templates = []
    entities  = []
    lines = open(file_name, "r").readlines()[1:]
    for line in lines:
        line = line.strip()
        parts= line.split("\t")
        if float(parts[1]) < 0.5:
            continue
        tokens = parts[2].split()
        extracts= parts[3].split()
        template = []
        for token,keep in zip(tokens,extracts):
            if int(keep) == 1:
                template.append(token)
        template = ' '.join(template).strip()
        if any([x not in template for x in ['<X>','</X>','<Y>','</Y>']]):
            continue
        templates.append(template)
        
        str1 = template[template.index("<X>"):template.index("</X>")+4]
        str2 = template[template.index("<Y>"):template.index("</Y>")+4]
        entities.append([str1,str2])
    return templates, entities



def get_simple_templated_sentences(sentence_relation_annotations,\
        sentence_entity_annotations, sentence_causes, sentence_contains,\
        title_entities, sentence_file_names, metadata, embeddings):

    healthline_graphs = []
    healthline_sentences = []
    for sentence in sentence_relation_annotations:
        output, _ = sentence_relation_annotations[sentence]
        data_points = output[0]
        for data_point in output[4]:
            data_points.append(data_point + ['contains'])
        healthline_graphs.append(get_causes_dict(data_points,\
        sentence_entity_annotations[sentence][0]))
        healthline_sentences.append(sentence)

    f = open("summary_train_template_sentences.txt", "w")
    f_contains = open("summary_train_template_sentences_contains.txt", "w")

    label_indices   = {}
    label_entities  = {}

    label_templates = {}
    label_entities  = {}

    for ind,graph in enumerate(healthline_graphs):
        if len(graph.keys()) == 1 and sum([len(graph[x]) for x in graph]) == 1\
            and len(sentence_entity_annotations[healthline_sentences[ind]][0].keys()) == 2:
            if metadata[sentence_file_names[healthline_sentences[ind]]]\
                    ['split'] == 'train':
                rev_entities = sentence_entity_annotations\
                        [healthline_sentences[ind]][0]
                entities = {x:y for y,x in rev_entities.items()}
                entity_names = list(rev_entities.keys())
                index1 = healthline_sentences[ind].index(entity_names[0])
                index2 = healthline_sentences[ind].index(entity_names[1])
                if index1 < index2:
                    if rev_entities[entity_names[0]] != 'Condition':
                        updated_sentence = healthline_sentences[ind][:index1] + \
                            " <X> " + entity_names[0] + " </X> " + \
                            healthline_sentences[ind]\
                            [index1+len(entity_names[0]):index2] + " <Y> " + \
                            entity_names[1] + " </Y> " + \
                            healthline_sentences[ind]\
                            [index2+len(entity_names[1]):]
                        f.write("0\t0\t" + updated_sentence + "\n")
                        if rev_entities[entity_names[0]] != 'Condition' and \
                            rev_entities[entity_names[1]] != 'Condition' and \
                            rev_entities[entity_names[1]] != 'Food':
                            f_contains.write("0\t0\t" + updated_sentence + "\n")
                        #f.write(updated_sentence + "\n")
                    if rev_entities[entity_names[1]] != 'Condition':
                        updated_sentence = healthline_sentences[ind][:index1] + \
                                    " <Y> " + entity_names[0] + " </Y> " + \
                                    healthline_sentences[ind]\
                                    [index1+len(entity_names[0]):index2] + " <X> " + \
                                    entity_names[1] + " </X> " + \
                                    healthline_sentences[ind]\
                            [index2+len(entity_names[1]):]
                        f.write("0\t0\t" + updated_sentence + "\n")
                        #f.write(updated_sentence + "\n")
                        if rev_entities[entity_names[0]] != 'Condition' and \
                                rev_entities[entity_names[1]] != 'Condition'\
                                and rev_entities[entity_names[0]] != 'Food':
                            f_contains.write("0\t0\t" + updated_sentence + "\n")

                else:
                    if rev_entities[entity_names[1]] != 'Condition':
                        updated_sentence = healthline_sentences[ind][:index2] + \
                            " <X> " + entity_names[1] + " </X> " + \
                            healthline_sentences[ind]\
                            [index2+len(entity_names[1]):index1] + " <Y> " + \
                            entity_names[0] + " </Y> " + \
                            healthline_sentences[ind]\
                            [index1+len(entity_names[0]):]
                        f.write("0\t0\t" + updated_sentence + "\n")
                        #f.write(updated_sentence + "\n")
                        if rev_entities[entity_names[0]] != 'Condition' and \
                            rev_entities[entity_names[1]] != 'Condition'\
                            and rev_entities[entity_names[0]] != 'Condition':
                            f_contains.write("0\t0\t" + updated_sentence + "\n")
                    if rev_entities[entity_names[0]] != 'Condition':
                        updated_sentence = healthline_sentences[ind][:index2] + \
                            " <Y> " + entity_names[1] + " </Y> " + \
                            healthline_sentences[ind]\
                            [index2+len(entity_names[1]):index1] + " <X> " + \
                            entity_names[0] + " </X> " + \
                            healthline_sentences[ind]\
                            [index1+len(entity_names[0]):]
                        f.write("0\t0\t" + updated_sentence + "\n")
                        if rev_entities[entity_names[0]] != 'Condition' and \
                            rev_entities[entity_names[1]] != 'Condition'\
                            and rev_entities[entity_names[1]] != 'Condition':
                            f_contains.write("0\t0\t" + updated_sentence + "\n")
                        #f.write(updated_sentence + "\n")


            label_indices[list(graph.keys())[0]] = \
                    label_indices.setdefault(list(graph.keys())[0],[]) + [ind]
            label_entities[healthline_sentences[ind]] = \
                sentence_relation_annotations[healthline_sentences[ind]][0]\
                [0][0] if \
                sentence_relation_annotations[healthline_sentences[ind]][0][0][0]\
                is not [] else \
                sentence_relation_annotations[healthline_sentences[ind]][0][4][0]

    f.close()
    f_contains.close()
    copyfile("summary_train_template_sentences.txt",\
            "summary_train_template_sentences_increases.txt")
    copyfile("summary_train_template_sentences.txt",\
            "summary_train_template_sentences_decreases.txt")
    copyfile("summary_train_template_sentences.txt",\
            "summary_train_template_sentences_controls.txt")
    copyfile("summary_train_template_sentences.txt",\
            "summary_train_template_sentences_satisfies.txt")

    label_templates['increases'], label_entities['increases'] = \
        get_rationale_entities(''\
                    'summary_train_template_sentences_increases.txt.'\
                    'rationale.machine_readable.tsv')
    label_templates['decreases'], label_entities['decreases'] = \
                    get_rationale_entities(''\
                    'summary_train_template_sentences_decreases.txt.'\
                    'rationale.machine_readable.tsv')
    label_templates['controls'], label_entities['controls'] = \
                    get_rationale_entities(''\
                    'summary_train_template_sentences_controls.txt.'\
                    'rationale.machine_readable.tsv')
    label_templates['satisfies'], label_entities['satisfies'] = \
                    get_rationale_entities(''\
                    'summary_train_template_sentences_satisfies.txt.'\
                    'rationale.machine_readable.tsv')
    label_templates['contains'], label_entities['contains'] = \
                    get_rationale_entities(''\
                    'summary_train_template_sentences_contains.txt.'\
                    'rationale.machine_readable.tsv')


    modified_sentences = {}
    templated_scores = []
    output_file = jsonlines.open("templated_multi_summaries.jsonl","w")
    input_sentiment_file = jsonlines.open("test1_sentiment.jsonl","w")
    output_sentiment_file = jsonlines.open("test2_sentiment.jsonl","w")
    for sentence in sentence_causes:
        entity_types = {}
        for entity in title_entities[sentence_file_names[sentence]]:
            entity_types[entity] = 'Food'
        if metadata[sentence_file_names[sentence]]['split'] != split_name:
            all_relations = sentence_causes[sentence]
            for contain_relation in sentence_contains[sentence]:
                all_relations.append(contain_relation + ['contains'])
        else:
            all_relations = metadata[sentence_file_names[sentence]]\
                    ['summary_inputs']\
                    ['summary_healthline_relation_annotations'][sentence.strip()][0]
        unique_relations = []
        added_relations = set()
        for relation in all_relations:
            if tuple([x.lower() for x in relation]) in added_relations:
                continue
            unique_relations.append(relation)
            added_relations.add(tuple([x.lower() for x in relation]))
        all_relations = unique_relations
        if len(all_relations) == 0:
            continue
        dict = {}
        dict['file_name'] = sentence_file_names[sentence]
        dict['gold'] = sentence
        dict['outputs'] = []
        dict['output']  = ''
        dict['ouput_dictionary'] = all_relations
        dict['split']  = metadata[sentence_file_names[sentence]]['split']
        dict1 = {'gold_label':'Good','sentence':sentence}
        for relation in all_relations:
            cause_type = relation[2]
            cosine_score = -1e100
            best_modified_sentence = ""
            if False:
                for template_index in label_indices.get(cause_type,[]):
                    modified_sentence = get_modified_sentence(\
                            sentence_relation_annotations,\
                            sentence_entity_annotations,\
                            healthline_sentences[template_index],\
                            [relation],\
                            {**sentence_entity_annotations[sentence][1],\
                            **entity_types}, sentence)
                    if relation[0].lower() in modified_sentence and \
                            relation[1].lower() in modified_sentence:
                        current_cosine = np.dot(_get_entity_embeddings(\
                                relation[0],embeddings),\
                                _get_entity_embeddings(\
                                label_entities[healthline_sentences[template_index]][0]\
                                ,embeddings)) + \
                                np.dot(_get_entity_embeddings(\
                                relation[1],embeddings),\
                                _get_entity_embeddings(\
                                label_entities[healthline_sentences[template_index]][1]\
                                ,embeddings))
                        if current_cosine > cosine_score:
                            cosine_score = current_cosine
                            best_modified_sentence = modified_sentence
            if True:
                for template, original_entities in zip(\
                        label_templates[cause_type], label_entities[cause_type]):
                    x_substr = template[template.index('<X>'):template.index('</X>')+4]
                    y_substr = template[template.index('<Y>'):template.index('</Y>')+4]
                    if len(template) <= len(x_substr) + len(y_substr) + 3:
                        continue
                    if x_substr == "" or y_substr == "":
                        continue
                    modified_sentence = template.replace(x_substr, relation[0])\
                            .replace(y_substr, relation[1])
                    if len(modified_sentence) <= len(relation[0]) +\
                            len(relation[1]) + 3:
                        continue
                    current_cosine = np.dot(_get_entity_embeddings(\
                                            relation[0],embeddings),\
                                            _get_entity_embeddings(\
                                            original_entities[0]\
                                            ,embeddings)) + \
                                            np.dot(_get_entity_embeddings(\
                                            relation[1],embeddings),\
                                            _get_entity_embeddings(\
                                            original_entities[1]\
                                            ,embeddings))\
                                    - len(modified_sentence.split())/15
                    if current_cosine > cosine_score:
                        cosine_score = current_cosine
                        best_modified_sentence = modified_sentence

            modified_sentence = best_modified_sentence
            #if relation[0].lower() not in modified_sentence or \
            #        relation[1].lower() not in modified_sentence:
            #    continue
            #relation1_index = modified_sentence.index(relation[0].lower())
            #relation2_index = modified_sentence.index(relation[1].lower())
            #if relation1_index < relation2_index:
            #    modified_sentence = modified_sentence[relation1_index:\
            #            relation2_index+len(relation[1])]
            #else:
            #    modified_sentence = modified_sentence[relation2_index:\
            #            relation1_index+len(relation[0])]
            dict['outputs'].append(modified_sentence)
        dict2 = {'gold_label':'Good','sentence':\
                ' '.join(dict['output']).strip()}
        dict['output'] = ' '.join(dict['outputs']).strip()
        output_file.write(dict)
        input_sentiment_file.write(dict1)
        output_sentiment_file.write(dict2)
    output_file.close()
    input_sentiment_file.close()
    output_sentiment_file.close()



if __name__ == "__main__":

    if os.path.exists("contains_wrong.jsonl"):
        os.remove("contains_wrong.jsonl")
        os.remove("contains_correct.jsonl")
    if os.path.exists("causes_wrong.jsonl"):
        os.remove("causes_wrong.jsonl")
        os.remove("causes_correct.jsonl")

    parser = argparse.ArgumentParser(description='Read features.')
    parser.add_argument('--run_features', action='store_true')
    parser.add_argument('--split_name', default='train', type=str)
    parser.add_argument('--fusion_model', default=\
            'transformer-abstractive-summarization/fusion/'\
            'discofuse_v1/wikipedia/classifier_large/pytorch_model.bin',\
            type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--repeat_instance', default=5, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--data_points', default=-1, type=int)
    parser.add_argument('--load_model', default='choice_policy.pt', type=str)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--task', default='policy', type=str)
    args = parser.parse_args()

    metadata = json.load(open("annotated_metadata5.json","r"))
    embeddings = read_embeddings()
    #fusion_state_dict = torch.load(args.fusion_model)
    #fusion_model = \
    #        BertForSequenceClassification.from_pretrained(\
    #        "bert-base-uncased", state_dict=fusion_state_dict, num_labels=5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",\
            do_lower_case=True)
    #fusion_model.cuda()
    fusion_model = None
    label_list = ["","and","but","although","however"]

    categorized_sentences = pickle.load(open("categorized_summaries.p","rb"))

    #policy = Policy(2)
    #policy = Policy(4)
    policy = PolicyChoices(2+2+2)
    if os.path.exists(args.load_model) and args.split_name != 'train':
        print("Loading saved model %s" %args.load_model)
        policy.load_state_dict(torch.load(open(args.load_model,"rb")))
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    target_entities = get_target_entities(metadata)
    target_embeddings= _get_all_entity_embeddings(target_entities, embeddings)

    sentence_relation_annotations, sentence_file_names, \
        sentence_entity_annotations,\
        sentence_modifier_entity_annotations,\
        sentence_relation_sentiments = paired_annotations(metadata)


    pickle.dump(sentence_relation_annotations, open(""\
            "sentence_relation_annotations.p","wb"))
    pickle.dump(sentence_file_names, open(""\
            "sentence_file_names.p","wb"))

    if (args.split_name == 'train' and args.run_features )\
            or (not os.path.exists("sentence_extracted_rationales.p")):
        get_property_rationales(metadata, sentence_relation_annotations,\
                sentence_file_names, sentence_entity_annotations, \
                sentence_relation_sentiments, splits=['train'])
        sentence_extracted_rationales = get_predicted_property_rationales()
        pickle.dump(sentence_extracted_rationales, open("\
            sentence_extracted_rationales.p", "wb"))
    else:
        sentence_extracted_rationales = \
                pickle.load(open("sentence_extracted_rationales.p", "rb"))


    property_style_rationales = {}
    for sentence in sentence_extracted_rationales:
        for property in sentence_extracted_rationales[sentence]:
            property_style_rationales[property] = \
                property_style_rationales.setdefault(property,[]) + \
                sentence_extracted_rationales[sentence][property]

    cause_style_rationales = \
            [x[x.find("###")+3:].strip() for x in \
            sum([sentence_extracted_rationales[x].get(\
            'causes',[]) for x in sentence_extracted_rationales],[])]


    title_entities = get_title_entities(metadata)

    if args.split_name == 'train' and args.run_features:
        get_fusion_training_data(sentence_extracted_rationales,\
            sentence_relation_annotations, sentence_file_names, \
            title_entities, embeddings)

    print("Found %d sentences with annotations paired." \
            %len(sentence_relation_annotations))

    sentence_population_relations, \
            sentence_population_entities = get_population_annotations(\
            sentence_relation_annotations,\
            sentence_modifier_entity_annotations)
    sentence_sentiment                   = get_sentiment_annotations(\
            metadata)


    old_keys = list(sentence_population_relations.keys())
    sentence_population_relations_a = {}
    sentence_population_entities_a = {}
    sentence_relation_annotations_a = {}
    for x in old_keys:
        sentence_population_relations_a[x.replace(' ','')] = \
                sentence_population_relations[x]
    old_keys = list(sentence_population_entities.keys())
    for x in old_keys:
        sentence_population_entities_a[x.replace(' ','')] = \
                sentence_population_entities[x]
    old_keys = list(sentence_relation_annotations.keys())
    for x in old_keys:
        sentence_relation_annotations_a[x.replace(' ','')] = \
                sentence_relation_annotations[x]

    sentence_pubmed_articles = {}
    for file_name in metadata:
        if 'summary_inputs' not in metadata[file_name]:
            continue
        if 'summary_pubmed_articles' not in metadata[file_name]\
            ['summary_inputs']:
            continue
        for sentence,pubmeds in metadata[file_name]\
            ['summary_inputs']['summary_pubmed_articles'].items():
            sentence_pubmed_articles[sentence.strip().replace(' ','')] = pubmeds

    pubmed_new_causes = pickle.load(open("pubmed_causes.p","rb"))


    pubmed_entity_types = get_corpus_entity_types(metadata)

    #get_population_correlation(sentence_population_entities,sentence_sentiment)
    #get_sentiment_statistics(sentence_sentiment, sentence_file_names,\
    #    metadata, ['train','dev','test'])
    #get_sentiment_statistics(sentence_sentiment, sentence_file_names,\
    #        metadata, ['train'])
    #get_sentiment_statistics(sentence_sentiment, sentence_file_names,\
    #        metadata, ['dev'])
    #get_sentiment_statistics(sentence_sentiment, sentence_file_names,\
    #        metadata, ['test'])

    sentence_causes, sentence_contains,\
            sentence_all_causes, sentence_all_contains,\
            gold_sentence_causes, gold_sentence_contains,\
            = follow_up_annotations(\
            sentence_relation_annotations,\
            embeddings, target_embeddings, sentence_file_names,\
            title_entities)

    pickle.dump(sentence_causes, open("sentence_causes.p","wb"))
    pickle.dump(sentence_contains,open("sentence_contains.p","wb"))
    pickle.dump(sentence_relation_annotations,open("\
            sentence_relation_annotations.p","wb"))


    split_name = args.split_name
    if split_name == 'train':

        vectorizer\
                = get_mapped_cosine_similarities(metadata,\
                        sentence_extracted_rationales, sentence_file_names,\
                        sentence_all_causes, embeddings)
        pickle.dump(vectorizer,open("vectorizer.p","wb"))

    else:
        vectorizer = pickle.load(open("vectorizer.p","rb"))
        sentence_extracted_rationales = pickle.load(open(\
                "sentence_extracted_rationales.p","rb"))

    if split_name == 'train':
        create_importance_classification_data(sentence_file_names, metadata,\
            sentence_causes, sentence_all_causes, sentence_contains,\
            sentence_all_contains, split_name)
    if False:
        sentence_structures = get_predicted_structures("importance_test.jsonl",\
                "/data/rsg/nlp/darsh/"+\
                "pytorch-pretrained-BERT/importance_classification/preds.jsonl",\
                embeddings, sentence_file_names, title_entities,\
                sentence_relation_sentiments, metadata,\
                cluster_threshold=1.5, prob_threshold=0.4)
        sentence_causes, sentence_contains = get_causes_contains_structures(\
                sentence_structures, sentence_all_causes, sentence_all_contains)
    else:
        input_sentences, output_sentences = \
                predict_importance_sentences(metadata,split_name,args)
        assert input_sentences is not None
        assert output_sentences is not None
        assert sentence_all_causes is not None
        assert sentence_all_contains is not None
        assert sentence_file_names is not None
        assert metadata is not None
        sentence_causes, sentence_contains = get_causes_contains_from_pubmed(\
                input_sentences, output_sentences, sentence_all_causes,\
                sentence_all_contains, sentence_file_names, metadata,\
                sentence_sentiment, vectorizer, embeddings, \
                gold_sentence_causes, policy, optimizer, args,\
                sentence_extracted_rationales,\
                fusion_model, tokenizer, label_list,\
                pubmed_entity_types,\
                property_style_rationales,\
                cluster_threshold=1.5, split_name=split_name)
    get_simple_templated_sentences(sentence_relation_annotations,\
            sentence_entity_annotations, sentence_causes, sentence_contains,\
            title_entities, sentence_file_names, metadata, embeddings)
    #causes,contains = compare_annotations(sentence_relation_annotations,\
    #        sentence_entity_annotations, embeddings,\
    #        target_embeddings, sentence_file_names, title_entities)
    #print(sum(causes)/len(causes),sum(contains)/len(contains))
