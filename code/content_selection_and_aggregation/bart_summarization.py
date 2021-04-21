import pdb
import json
import pickle
import numpy as np
import re
from get_model_performances import _get_entities
from analysis.map_condition_phrases import read_embeddings
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import jsonlines
import os
random.seed(42)

num_instances = 1

def _get_pointer_data(text):

    if '|SEN|' in text:
        text = text.split('|SEN|')[1].strip()
    text = text[text.find(">")+1:text.rfind("<")].strip()
    parts= text.split("<")
    parts= ["Food " + x.strip() for x in parts[:1]] + \
            ["<"+x.strip() for x in parts[1:]]
    return parts

def _get_pubmed_sentences():

    metadata = json.load(open("annotated_metadata5.json","r"))
    pubmed_sentences = {}
    for file_name in metadata:
        if 'pubmed_sentences' not in metadata[file_name]:
            continue
        for pubmed in metadata[file_name]['pubmed_sentences']:
            text = ""
            for tuple in \
                    metadata[file_name]['pubmed_sentences'][pubmed]:
                sentence = tuple[0]
                text += ' '.join(sentence).strip() + ' '
            pubmed_sentences[pubmed] = text
    return pubmed_sentences

def _get_pubmed_causes():

    metadata = json.load(open("annotated_metadata5.json","r"))
    pubmed_causes = {}
    for pubmed in metadata['pubmed_sentences_annotations']:
        if 'pubmed_sentences_relation_annotations' not in \
                metadata['pubmed_sentences_annotations'][pubmed]:
            continue
        pubmed_causes[pubmed] = metadata['pubmed_sentences_annotations']\
                [pubmed]['pubmed_sentences_relation_annotations'][0]
    return pubmed_causes

def _get_new_pubmed_causes():

    pubmed_causes = pickle.load(open("pubmed_causes.p","rb"))
    return pubmed_causes

def _get_kl(text, produced_lengths):

    if '|SEN|' in text:
        text = text[text.find("|SEN|")+5:].strip()
    keep_generate = [1 - int('<' in x and '>' in x)\
        for x in text.split()]
    assert len(produced_lengths) == keep_generate.count(0)
    p_lengths = []
    for i in range(len(keep_generate)):
        if keep_generate[i] == 0:
            p_lengths.append(\
                    produced_lengths[keep_generate[:i].count(0)])
        else:
            p_lengths.append(0)
    return keep_generate, p_lengths


def _preprocess_string(text):
    text = ' '.join(word_tokenize(text)).strip()
    if '\\' in text:
        text = text.replace('\\','')
    if 'vitamin e.' in text:
        text = text.replace('vitamin e.', 'vitamin e .')
    if 'and e.' in text:
        text = text.replace('and e.', 'and e .')
    if 'xe2x80x99' in text:
        text = text.replace('xe2x80x99',' \'')
    if '-' in text:
        text = text.replace('-',' - ')
        text = ' '.join(text.split()).strip()
    return text


def get_splits_and_pubmeds(sentences):

    metadata = json.load(open("annotated_metadata5.json","r"))
    sentence_split_pubmed = {}
    sentences_modified = [sentence.replace(' ','') for sentence in sentences]
    for file_name in metadata:
        if 'summary_inputs' not in metadata[file_name]:
            continue
        if 'summary_pubmed_articles' not in metadata[file_name]\
            ['summary_inputs']:
            continue
        for sentence,pubmeds in metadata[file_name]\
            ['summary_inputs']['summary_pubmed_articles'].items():
            if sentence.replace(' ','') not in sentences_modified:
                continue
            index = sentences_modified.index(sentence.replace(' ',''))
            sentence_split_pubmed[sentences[index]] = [\
                    metadata[file_name]['split'], pubmeds, file_name]
    return sentence_split_pubmed


def score_uncategorized_triplets(pubmeds, pubmed_cause_lists, \
        pubmed_sentences, key):

    text_counters = {}
    for pubmed in pubmeds:
        counters    = []
        cause_lists = []
        causes_list = pubmed_cause_lists[pubmed]
        for cause in causes_list:
            if pubmed not in pubmed_sentences or cause[2] == 'None':
                count = 0
            else:
                count   = pubmed_sentences[pubmed].count(cause[0]) + \
                    pubmed_sentences[pubmed].count(cause[1])
            counters.append(count)
            cause_lists.append(cause[-1])
            text_counters[cause[-1]] = count
    return text_counters

def score_categorized_triplets(pubmeds, pubmed_cause_lists, \
        pubmed_sentences, key):

    text_counters = {}
    for pubmed in pubmeds:
        counters  = []
        cause_lists = []
        causes_list = pubmed_cause_lists[pubmed]
        for cause in causes_list:
            if pubmed not in pubmed_sentences or (cause[2] == 'None' and\
                    key=='certain') or (cause[2] !='unclear/insignificant'\
                    and key!='certain'):
                count = -100 if pubmed not in pubmed_sentences \
                        or cause[2] == 'None' else -100 + \
                        pubmed_sentences[pubmed].count(cause[0]) + \
                        pubmed_sentences[pubmed].count(cause[1])
            else:
                count   = pubmed_sentences[pubmed].count(cause[0]) + \
                        pubmed_sentences[pubmed].count(cause[1])
            text_counters[cause[-1]] = count
    return text_counters

def all_texts(pubmeds, pubmed_cause_lists, pubmed_sentences, key):

    text = ""
    for pubmed in pubmeds:
        text += pubmed_sentences.get(pubmed,"") + " "
    text = text.strip()
    return ' '.join(text.split()[:1000]).strip()

def get_uncategorized_parallel_sentences(categorized_sentence_splits,\
        add_source=False,scoring_function = score_uncategorized_triplets,\
        vanilla=False):

    pubmed_causes = pickle.load(open("pubmed_causes.p","rb"))
    pubmed_sentences = _get_pubmed_sentences()

    file_names    = {split:{'source':split+".source",'target':split+".target"}\
            for split in ['train','dev','test']}
    file_pointers = {split:{'source':open(file_names[split]['source'],"w"),\
            'target':open(file_names[split]['target'],"w")} for split \
            in ['train','dev','test']}

    for key,sentence_splits in categorized_sentence_splits.items():
        for sentence in sentence_splits:
            split, pubmeds, file_name = sentence_splits[sentence]
            if scoring_function != all_texts:
                text_counters  = scoring_function(pubmeds,\
                        {p:pubmed_causes.get(p,[]) for p in pubmeds}, \
                        pubmed_sentences, key)
                top_texts      = [t for t in \
                        sorted(text_counters,key=text_counters.get,\
                        reverse=True)[:5]]
                top_texts      = [file_name] + top_texts
                if len(top_texts) == 0:
                    continue
                for k in categorized_sentence_splits.keys():
                    if k!=key: #and split != 'test' and split=='test':
                        continue
                    if not add_source:
                        file_pointers[split]['source'].write(\
                            ' '.join(top_texts).strip()+"\n")
                    else:
                        file_pointers[split]['source'].write(\
                            k+" # " +' '.join(top_texts).strip()+"\n")
                    file_pointers[split]['target'].write(sentence+"\n")
            else:
                all_text = scoring_function(pubmeds, None, \
                        pubmed_sentences, key)
                all_text += file_name + " # " + all_text
                file_pointers[split]['source'].write(all_text+"\n")
                file_pointers[split]['target'].write(sentence+"\n")

    for split in file_pointers:
        file_pointers[split]['source'].close()
        file_pointers[split]['target'].close()

def get_representation(term, embeddings):

    array = np.array([0.0 for _ in range(50)])
    words = []
    if ' ' in  term:
        words = term.split()
    else:
        words = [term]
    for word in words:
        array += embeddings.get(word,np.array([0.0 for _ in range(50)]))
    if np.linalg.norm(array)>0.0:
        array /= np.linalg.norm(array)
    return array


def get_closest_terms(terms, embeddings):
    term_embeddings = np.vstack([get_representation(term, embeddings)\
            for term in terms])
    term_closest = {}
    for i in range(len(terms)):
        dot_products = np.dot(term_embeddings[i],term_embeddings.T)
        array_indices= np.argsort(-dot_products)[1:]
        term_closest[terms[i]] = [terms[j] for j in array_indices]
    return term_closest


def create_bart_infilling_data(add_source=False):

    embeddings = read_embeddings()
    all_causes = []
    all_foods  = []
    all_nutritions = []

    pubmed_sentences = _get_pubmed_sentences()
    f = open("pretrained.lm.source","w")
    for text in pubmed_sentences.values():
        sentences = sent_tokenize(text)
        for sentence in sentences:
            f.write(sentence + "\n")
    f.close()

    sentence_causes = pickle.load(open("sentence_causes.p","rb"))
    sentence_contains = pickle.load(open("sentence_contains.p","rb"))

    for sentence in sentence_causes:
        for cause in sentence_causes[sentence]:
            all_causes.append(cause[1])
    for sentence in sentence_contains:
        for cause in sentence_contains[sentence]:
            all_foods.append(cause[0])
            all_nutritions.append(cause[1])

    all_foods = list(set(all_foods))
    all_causes = list(set(all_causes))
    all_nutritions = list(set(all_nutritions))

    food_closest = get_closest_terms(all_foods,embeddings)
    causes_closest = get_closest_terms(all_causes,embeddings)
    nutrition_closest = get_closest_terms(all_nutritions,embeddings)

    metadata = json.load(open("annotated_metadata5.json","r"))
    good_cases = 0
    bad_cases  = 0
    t_source_file = open("train.source","w")
    t_expanded_source_file = open("train_expanded.source","w")
    t_target_file = open("train.target","w")
    t_k_file      = open("train_k.source","w")
    t_l_file      = open("train_l.source","w")
    t_lm_source   = open("train.lm.source","w")
    d_source_file = open("dev.source","w")
    d_target_file = open("dev.target","w")
    d_k_file      = open("dev_k.source","w")
    d_l_file      = open("dev_l.source","w")
    d_lm_source   = open("dev.lm.source","w")
    test_gold     = open("test.gold","w")

    full_pointer_data = []

    categorized_summaries = pickle.load(open("categorized_output_summaries.p",\
            "rb"))
    #categorized_pubmed_summaries = pickle.load(open(\
    #        "categorized_pubmed_summaries.p","rb"))
    categorized_pubmed_summaries = {}

    categorized_input_summaries = pickle.load(open(\
            "categorized_input_summaries.p","rb"))

    summaries_categories  = {}
    for category,sentences in categorized_summaries.items():
        for sentence in sentences:
            summaries_categories[sentence] = category
    for category in categorized_pubmed_summaries:
        #['contradiction','not_enough']:
        for sentence in categorized_pubmed_summaries[category]:
            summaries_categories[sentence] = category

    input_categorized_summaries = {}
    for category in categorized_input_summaries:
        for sentence in categorized_input_summaries[category]:
            input_categorized_summaries[sentence] = category

    healthline_entities = pickle.load(open("healthline_entities.p","rb"))
    healthline_causes   = pickle.load(open("healthline_causes.p","rb"))

    found = 0
    not_found = 0
    total = 0
    not_added_count = 0
    sentences_considered = set()

    sentence_relation_annotations = {}
    sentence_population_annotations   = {}

    categorized_counts = {}

    for file_name in tqdm(metadata):
        if 'split' not in metadata[file_name]:
            continue
        if metadata[file_name]['split'] == 'test':
            continue
        if 'summary_inputs' not in metadata[file_name]:
            continue
        if 'summary_healthline_relation_annotations' not in \
                metadata[file_name]['summary_inputs']:
            continue
        sentence_tuples = metadata[file_name]['summary_inputs']\
                ['summary_healthline_entity_annotations']
        for s_t in sentence_tuples:
            sentence_population_annotations[s_t[0].replace(' ','')] = \
                    _get_entities(s_t[0].split(),s_t[1].split(),'Population')
        for Sentence, annotation in metadata[file_name]['summary_inputs']\
                ['summary_healthline_relation_annotations'].items():
        #for sentence_tuple in metadata[file_name]['summary_inputs']\
        #        ['summary_healthline_entity_annotations']:
        #    Sentence, entity_annotations = sentence_tuple
            sentence_relation_annotations[Sentence] = annotation
    for pubmed in tqdm(metadata['pubmed_sentences_annotations']):
        if 'pubmed_sentences_relation_annotations' not \
                in metadata['pubmed_sentences_annotations'][pubmed] or\
            'pubmed_sentences_entity_annotations' not \
            in metadata['pubmed_sentences_annotations'][pubmed]:
            continue
        meta_annotation  = metadata['pubmed_sentences_annotations']\
                [pubmed]['pubmed_sentences_relation_annotations']
        for sentence_tuple in metadata['pubmed_sentences_annotations']\
                [pubmed]['pubmed_sentences_entity_annotations']:
            sentence = sentence_tuple[0]
            annotation = []
            for j in range(10):
                annotation.append([])
            for ann in meta_annotation[0]:
                if ann[0] in sentence and ann[1] in sentence:
                    annotation[0].append(ann)
            for ann in meta_annotation[4]:
                if ann[0] in sentence and ann[1] in sentence:
                    annotation[4].append(ann)
            sentence_relation_annotations[sentence] = annotation
    if len(sentence_relation_annotations) > 0:
        for Sentence, annotation in sentence_relation_annotations.items():
            import random
            if random.random()>0.9:
                metadata[file_name]['split'] = 'dev'
            else:
                metadata[file_name]['split'] = 'train'
            total += 1
            populations = sentence_population_annotations.get(\
                    Sentence.replace(' ',''),[])
            sentences_considered.add(Sentence)
            cat_sentence = ' '.join(word_tokenize(Sentence.strip())).strip()
            if cat_sentence not in summaries_categories:
                not_found += 1
                continue
            #if "effects" in cat_sentence:
            #    continue
            found += 1
            bad = False
            sentence = Sentence.lower()
            if 'alzheimer\\\'s' in sentence:
                sentence = sentence.replace('alzheimer\\\'s','alzheimer \'s')
            sentence = _preprocess_string(sentence)
            causes = annotation[0]
            causes = [[_preprocess_string(x) for x in y] for y in causes]
            contains=annotation[4]
            contains = [[_preprocess_string(x) for x in y] for y in contains]
            correct_contains = []
            for contain in contains:
                if contain[0] == contain[1]:
                    continue
                correct_contains.append(contain)
            contains = correct_contains
            correct_causes = []
            for cause in causes:
                if cause[0] == cause[1]:
                    continue
                correct_causes.append(cause)
            causes = correct_causes
            exit   = False
            for cause in causes+contains:
                if cause[0] not in sentence:
                    exit = True
                    break
                if cause[1] not in sentence:
                    exit = True
                    break
            if exit:
                continue
            if summaries_categories[cat_sentence] == 'scoping':
                for pop in populations:
                    if len(causes)>0:
                        causes.append([causes[0][0].lower(),\
                                _preprocess_string(pop.lower()),'Population'])
            contain_max_index = 0
            sentence = ' '.join(word_tokenize(sentence)).strip()
            if '-' in sentence:
                sentence = sentence.replace('-',' - ')
                sentence = ' '.join(sentence.strip().split()).strip()
            for contain in contains:
                if sentence.index(contain[0]) > sentence.index(contain[1]):
                    bad = True
                    break
                contain_max_index = sentence.index(contain[1])
            for cause in causes:
                if sentence.index(cause[0]) > sentence.index(cause[1]):
                    bad = True
                    break
                if contain_max_index > sentence.index(cause[1]):
                    bad = True
                    break
            if len(set([c[0] for c in contains])) > 1:
                bad = True
            if len(causes) == 0 and len(contains) == 0:
                bad = True
            if bad:
                bad_cases += 1
            else:
                good_cases += 1
            if bad:
                continue
            input_text = ""
            assert len(set([c[0] for c in contains])) <= 1
            contains = list(set([tuple(x) for x in contains]))
            indices = [sentence.index(c[1]) for c in contains]
            indices = np.argsort(indices)
            contains = [contains[i] for i in indices]
            indices = [sentence.index(c[1]) for c in causes]
            indices = np.argsort(indices)
            sorted_causes = [causes[i] for i in indices]
            foods   = []
            conditions = []
            nutritions = []
            if len(contains) > 0:
                input_text = "<blank> " + contains[0][0] + " <contains> "
                foods.append(contains[0][0])
                for c in contains[:len(contains)-1]:
                    if c[1] in input_text:
                        continue
                    input_text += c[1] + " <contains> "
                    nutritions.append(c[1])
                input_text += contains[-1][1]
                nutritions.append(contains[-1][1])
            else:
                input_text = "<blank> " + causes[0][0]
                foods.append(causes[0][0])
            for cause in sorted_causes:
                if cause[1] in input_text:
                    continue
                input_text += " <" + cause[2] + "> " + cause[1]
                conditions.append(cause[1])
            input_text += " <blank>"
            if input_text.count("<") == 1:
                continue
            s_indices     = [m.start() for m in re.finditer('<', input_text)]\
                    [1:]
            e_indices     = [m.start() for m in re.finditer('>', input_text)]
            good_output = ""
            start_ind   = 0
            end_ind     = -1
            produced_lengths= []
            keep_generate   = []
            for s,e in zip(s_indices, e_indices):
                find_str = input_text[e+1:s].strip()
                find_ind = sentence.index(find_str)
                good_output += " # " + sentence[start_ind:find_ind].strip()
                if ' ' not in sentence[start_ind:find_ind].strip():
                    produced_lengths.append(int(\
                            sentence[start_ind:find_ind].strip()!=''))
                else:
                    produced_lengths.append(len(sentence[start_ind:\
                            find_ind].strip().split()))
                start_ind= find_ind + len(find_str)
            good_output += " # " + sentence[start_ind:].strip()
            if ' ' not in sentence[start_ind:].strip():
                produced_lengths.append(int(sentence[start_ind:].strip()!=''))
            else:
                produced_lengths.append(len(sentence[start_ind:\
                        ].strip().split()))
            if max(produced_lengths) > 25:
                continue
            good_output  = good_output.strip()
            keep_generate, p_lengths = _get_kl(input_text, \
                    produced_lengths)
            #print(input_text, good_output, sentence)
            if not (len(sentence.split()) == \
                    sum(p_lengths)+keep_generate.count(1)):
                continue
            name_counter = {'certain':'$1$', 'not_enough':'$2$',\
                        'scoping':'$3$', 'contradiction':'$4$'}
            target_text_replacement = {'$1$':'Certain','$2$':'Not Enough',\
                    '$3$':'Scoping','$4$':'Contradiction'}
            multiplier = {'certain':0,'not_enough':6,'scoping':80,\
                    'contradiction':4000}
            if summaries_categories[cat_sentence] not in\
                    ['contradiction','scoping','not_enough','certain']:
                continue
            if summaries_categories[cat_sentence] == 'scoping':
                if 'Population' not in input_text:
                    continue
            if metadata[file_name]['split'] == 'train':
                if add_source:
                    #input_text = name_counter[summaries_categories\
                    #        [cat_sentence]] + \
                    #        ' |SEN| ' + input_text
                    input_text = random.choice(\
                            categorized_summaries[summaries_categories\
                            [cat_sentence]]) + \
                            ' |SEN| ' + input_text
                t_source_file.write(input_text + "\n")
                categorized_counts[summaries_categories[cat_sentence]] = \
                        categorized_counts.setdefault(\
                        summaries_categories[cat_sentence],0) + 1
                pointer_parts = _get_pointer_data(input_text)
                full_pointer_data.append(pointer_parts)
                if sentence.startswith(":"):
                    sentence = sentence[sentence.find(":")+1:].strip()
                if add_source:
                    for name in ["$2$","$4$"]:#name_counter.values():
                        t_expanded_source_file.write(name + input_text[2:] + "\n")
                #t_target_file.write(good_output + "\n")
                t_target_file.write(sentence + "\n")
                #t_target_file.write(target_text_replacement\
                #        [name_counter[summaries_categories[cat_sentence]]]+\
                #        "\n")
                t_k_file.write(" ".join([str(x) for x in keep_generate]\
                        ).strip()+"\n")
                t_l_file.write(" ".join([str(x) for x in p_lengths\
                        ]).strip()+"\n")
                t_lm_source.write(input_text + " |SEN| "+good_output + "\n")
                assert input_text.count("<") == good_output.count("#")
                for food in foods:
                    target_foods = food_closest.get(food,[])[:num_instances*\
                            multiplier[summaries_categories[cat_sentence]]]
                    for target_food in target_foods:
                        text = random.choice(categorized_summaries[summaries_categories\
                    [cat_sentence]]) + " |SEN| " + input_text.split("|SEN|")[1].replace(food,target_food)
                        assert good_output.count("#") == text.count("<")
                        keep_generate, p_lengths = _get_kl(text,\
                                produced_lengths)
                        if not (len(sentence.replace(food,target_food).split()) == \
                            sum(p_lengths)+keep_generate.count(1)):
                            continue
                        t_source_file.write(text + "\n")
                        pointer_parts = _get_pointer_data(text)
                        full_pointer_data.append(pointer_parts)
                        if add_source:
                            for name in ["$2$","$4$"]:#name_counter.values():
                                t_expanded_source_file.write(name + text[2:] + "\n")
                        t_k_file.write(" ".join([str(x) for x in keep_generate]\
                                ).strip()+"\n")
                        t_l_file.write(" ".join([str(x) for x in p_lengths\
                                ]).strip()+"\n")
                        #t_target_file.write(good_output + "\n")
                        t_target_file.write(sentence.replace(\
                                food,target_food) + "\n")
                        categorized_counts[summaries_categories[cat_sentence]] = \
                                    categorized_counts.setdefault(\
                                    summaries_categories[cat_sentence],0) + 1
                        #t_target_file.write(target_text_replacement\
                        #        [name_counter[summaries_categories\
                        #        [cat_sentence]]]+"\n")
                        t_lm_source.write(text + " |SEN| "+good_output + "\n")
                for nutrition in nutritions:
                    target_nutritions = \
                            nutrition_closest.get(nutrition,[])[:num_instances*\
                            multiplier[summaries_categories[cat_sentence]]]
                    for target_nutrition in target_nutritions:
                        text = random.choice(categorized_summaries[summaries_categories\
                            [cat_sentence]]) + " |SEN| " + input_text.split("|SEN|")[1].replace(nutrition,target_nutrition)
                        #text = input_text.replace(nutrition,\
                        #        target_nutrition)
                        keep_generate, p_lengths = _get_kl(text,\
                                                                produced_lengths)
                        if not (len(sentence.replace(nutrition,target_nutrition).split()) == \
                            sum(p_lengths)+keep_generate.count(1)):
                            continue
                        pointer_parts = _get_pointer_data(text)
                        full_pointer_data.append(pointer_parts)
                        t_source_file.write(text + "\n")
                        if add_source:
                            for name in ["$2$","$4$"]:#name_counter.values():
                                t_expanded_source_file.write(name + text[2:] + "\n")
                        t_k_file.write(" ".join([str(x) for x in keep_generate]\
                                ).strip()+"\n")
                        t_l_file.write(" ".join([str(x) for x in p_lengths\
                                ]).strip()+"\n")
                        #t_target_file.write(good_output + "\n")
                        t_target_file.write(sentence.replace(nutrition,\
                                target_nutrition) + "\n")
                        categorized_counts[summaries_categories[cat_sentence]] = \
                                categorized_counts.setdefault(\
                        summaries_categories[cat_sentence],0) + 1
                        #t_target_file.write(target_text_replacement\
                        #        [name_counter[summaries_categories\
                        #        [cat_sentence]]]+"\n")
                        assert good_output.count("#") == text.count("<")
                        t_lm_source.write(text + " |SEN| "+good_output + "\n")
                for condition in conditions:
                    condition_nutritions = \
                            causes_closest.get(condition,[])[:num_instances*\
                            multiplier[summaries_categories[cat_sentence]]]
                    for condition_nutrition in condition_nutritions:
                        #text = input_text.replace(condition,\
                        #        condition_nutrition)
                        text = random.choice(categorized_summaries[summaries_categories\
                                [cat_sentence]]) + " |SEN| " + input_text.split("|SEN|")[1].replace(condition,\
                                condition_nutrition)
                        keep_generate, p_lengths = _get_kl(text,\
                                            produced_lengths)
                        if not (len(sentence.replace(condition,condition_nutrition).split()) == \
                            sum(p_lengths)+keep_generate.count(1)):
                            continue
                        pointer_parts = _get_pointer_data(text)
                        full_pointer_data.append(pointer_parts)
                        t_source_file.write(text + "\n")
                        if add_source:
                            for name in ["$2$","$4$"]:#name_counter.values():
                                t_expanded_source_file.write(name + text[2:] + "\n")
                        t_k_file.write(" ".join([str(x) for x in keep_generate]\
                                ).strip()+"\n")
                        t_l_file.write(" ".join([str(x) for x in p_lengths\
                                ]).strip()+"\n")
                        #t_target_file.write(good_output + "\n")
                        t_target_file.write(sentence.replace(condition,\
                                condition_nutrition) + "\n")
                        categorized_counts[summaries_categories[cat_sentence]] = \
                                categorized_counts.setdefault(\
                                summaries_categories[cat_sentence],0) + 1
                        #t_target_file.write(target_text_replacement\
                        #        [name_counter[summaries_categories\
                        #        [cat_sentence]]]+"\n")
                        assert good_output.count("#") == text.count("<")
                        t_lm_source.write(text + " |SEN| "+good_output + "\n")
            elif metadata[file_name]['split'] == 'dev':
                assert input_text.count("<") == good_output.count("#")
                if add_source:
                    for prefix in multiplier:
                    #for a in range(1):
                        #text = name_counter[summaries_categories\
                        #        [cat_sentence]] + " |SEN| " + input_text
                        text = random.choice(categorized_summaries[\
                                prefix]) + " |SEN| " + input_text
                        #name_counter[prefix] + \
                        #        " |SEN| " + input_text
                        d_source_file.write(text + "\n")
                        keep_generate, p_lengths = _get_kl(text,\
                            produced_lengths)
                        #d_target_file.write(good_output + "\n")
                        d_lm_source.write(text + " |SEN| "+good_output + "\n")
                        d_target_file.write(sentence + "\n")
                        #d_target_file.write(target_text_replacement[\
                        #        name_counter[prefix]]+"\n")
                        d_k_file.write(" ".join([str(x) for x in keep_generate]\
                            ).strip()+"\n")
                        d_l_file.write(" ".join([str(x) for x in p_lengths\
                            ]).strip()+"\n")

    t_source_file.close()
    t_expanded_source_file.close()
    t_target_file.close()
    t_k_file.close()
    t_l_file.close()
    d_source_file.close()
    d_target_file.close()
    d_k_file.close()
    d_l_file.close()
    t_lm_source.close()
    d_lm_source.close()
    pickle.dump(full_pointer_data,open("pointer_data.p","wb"))

    test_source_file = open("test.source","w")
    test_meteor_source_file = open("test_meteor.source","w")
    extractive_test_file = jsonlines.open(\
            "extractive_multi_summaries.jsonl","r")
    sentence_selected_causes = {}

    pubmed_causes = _get_new_pubmed_causes()#_get_pubmed_causes()

    metadata = json.load(open("annotated_metadata5.json","r"))

    data_points = 0

    test_pointer_data = []

    file_name_not_found = 0

    if os.path.exists("/data/rsg/nlp/darsh/pointer-networks-pytorch/"\
            "sorted_pointed_data.p"):
        sorted_pointed_data = pickle.load(open(\
            "/data/rsg/nlp/darsh/pointer-networks-pytorch/"\
            "sorted_pointed_data.p","rb"))
    else:
        sorted_pointed_data = {}

    all_considered = 0
    test_categories = {}
    scoping_populations = pickle.load(open("scoping_populations.p","rb"))

    for dicti in extractive_test_file:
        if dicti['consider_tuples'] == []:
            continue
        if dicti['file_name'] not in metadata:
            continue
        if 'summary_pubmed_articles' not in metadata[dicti['file_name']]\
                ['summary_inputs']:
            continue
        all_considered += 1
        dictionaries = metadata[dicti['file_name']]\
                ['summary_inputs']['summary_pubmed_articles']
        dictionaries = {x.strip():y for x,y in dictionaries.items()}
        pubmeds = dictionaries[dicti['gold']]
        causes = sum([pubmed_causes.get(x,[]) for x in pubmeds],[])
        causes_dict = {tuple(x[:2]):x[2] for x in causes if x[2]!='None' and\
                x[2]!='unclear/insignificant'}
        input_causes = []
        input_contains=[]
        for consider_tuple in dicti['consider_tuples']:
            if len(consider_tuple) > 2:
                if tuple(consider_tuple[:2]) not in causes_dict:
                    continue
                input_causes.append(tuple(consider_tuple[:2]\
                        + [causes_dict[tuple(consider_tuple[:2])]]))
            else:
                input_contains.append(tuple([x.strip() \
                        for x in consider_tuple[:2]]))
        if len(input_contains) == 0 and len(input_causes) == 0:
            continue
        sample_text = random.choice(categorized_summaries\
                [input_categorized_summaries[dicti['gold']]])
        test_categories[input_categorized_summaries[dicti['gold']]] = \
                test_categories.setdefault(input_categorized_summaries\
                [dicti['gold']],0) + 1
        if input_categorized_summaries[dicti['gold']] == 'scoping':
            if dicti['gold'] in scoping_populations:
                input_causes.append(["",\
                        scoping_populations[dicti['gold']],'Population'])
        data_points += 1
        selected_inputs = set()
        good_output = "<blank> "
        added_something = False

        for ind,input_contain in enumerate(input_contains):
            if input_contain[0] in dicti['file_name'].replace('-',' '):
                good_output += input_contain[0]
                added_something = True
                selected_inputs.add(input_contain[0])
                break

        if not added_something and dicti['food_name'] == '':
            continue

        if not added_something:
            good_output += dicti['food_name']
            added_something = True

        contains_added = set()

        trim_input_contains = []
        for ind,input_contain in enumerate(input_contains):
            dont_keep = False
            for ind2,input_contain2 in enumerate(input_contains):
                if ind==ind2:
                    continue
                if input_contain[1] in input_contain2[1]:
                    dont_keep = True
                    break
            if not dont_keep:
                trim_input_contains.append(input_contain)

        input_contains = trim_input_contains

        trim_input_causes = []
        for ind,input_cause in enumerate(input_causes):
            keep = True
            for ind2, input_cause2 in enumerate(input_causes):
                if ind==ind2:
                    continue
                if input_cause[1].lower() in input_cause2[1].lower():
                    keep = False
                    break
            if keep:
                trim_input_causes.append(input_cause)

        input_causes = trim_input_causes

        for input_contain in input_contains:
            if input_contain[0] in dicti['file_name'].replace('-',' ')\
                    or len(selected_inputs)==0:
                if input_contain[1].lower() in contains_added:
                    continue
                good_output += " <contains> " + input_contain[1]
                added_something = True
                selected_inputs.add(input_contain[1].lower())
                contains_added.add(input_contain[1].lower())

        #if not added_something:
            #good_output += dicti['food_name']
            #added_something = True
            #for input_cause in input_causes:
            #    if input_cause[0] in dicti['file_name'].replace('-',' '):
            #        good_output += input_cause[0]
            #        added_something = True
            #        selected_inputs.add(input_cause[0])
            #        break

        for input_cause in input_causes:
            if input_cause[0].lower() in selected_inputs or len(selected_inputs)==0:
                good_output += " <"+input_cause[2]+"> " + input_cause[1]
                added_something = True
        good_output += " <blank>"
        if not added_something:
            not_added_count += 1
            continue
        assert good_output.startswith("<blank>")
        assert good_output.endswith("<blank>")
        pointer_input = _get_pointer_data(\
                sample_text + " |SEN| " + good_output)
        pointer_output = sorted_pointed_data.get(tuple(pointer_input),\
                pointer_input)
        test_pointer_data.append(pointer_input)
        #print(pointer_output)
        #print(pubmeds)
        #print(" ".join(pointer_output).strip())
        #print(good_output)
        if "<" not in " ".join(pointer_output).strip():
            continue
        test_source_file.write(sample_text + " |SEN| " + \
                "<blank> " + " ".join(pointer_output).strip()[5:]\
                +" <blank>"+"\n")
        test_meteor_source_file.write(random.choice(categorized_summaries\
                ['certain'])+ " |SEN| " + \
                "<blank> " + " ".join(pointer_output).strip()[5:]\
                +" <blank>"+"\n")
        test_gold.write(dicti['gold']+"\n")


    print(data_points, file_name_not_found, not_added_count, all_considered)
    test_source_file.close()
    pickle.dump(test_pointer_data,open("test_pointer_data.p","wb"))
    test_gold.close()
    print(categorized_counts)
    print(test_categories)
    test_meteor_source_file.close()


def main():

    categorized_sentences = pickle.load(open("categorized_summaries.p","rb"))
    categorized_sentence_splits = {}
    for key in categorized_sentences.keys():
        categorized_sentence_splits[key] = \
                get_splits_and_pubmeds(categorized_sentences[key])
    #get_uncategorized_parallel_sentences(categorized_sentence_splits,\
    #        add_source=False,\
    #        scoring_function=all_texts)
    create_bart_infilling_data(add_source=True)


if __name__ == "__main__":
    main()
