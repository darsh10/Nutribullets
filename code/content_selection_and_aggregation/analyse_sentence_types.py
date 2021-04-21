import json
from tqdm import tqdm
import pdb
import pickle
import nltk
import numpy as np
from get_model_performances import  _get_entities, _get_entity_embeddings
from bart_summarization import get_closest_terms
from analysis.map_condition_phrases import read_embeddings
from evaluation.create_kg import create_entity_annotations_file, _perform_tagging, get_sentence_entities
from evaluation.create_kg import _read_entity_file, create_relation_annotations_file, produce_re_outputs, get_predicted_relations
import random
import jsonlines
from metapub import PubMedFetcher
random.seed(42)

def analyse_scoping(data, selected_so_far):

    summaries = []

    for file_name in tqdm(data):
        if 'summary_inputs' not in data[file_name]:
            continue
        if 'summary_healthline_entity_annotations' not in \
                data[file_name]['summary_inputs']:
            continue
        for sentence_tuple in data[file_name]['summary_inputs']\
                ['summary_healthline_entity_annotations']:
            summary = sentence_tuple[0]
            entities= sentence_tuple[1]
            if summary.strip() in selected_so_far or \
                    summary in selected_so_far or \
                    is_insufficient(summary):
                continue
            assert "more research" not in summary[0]
            populations = _get_entities(summary.split(),entities.split(),\
                    'Population')
            if len(set(populations)) != 1:
                continue
            if all([x not in populations[0] for x in \
                    ['human','rat','animal','cell','men','women',\
                    'adult humans']]):
                summaries.append(summary.strip())
    print(len(summaries))
    return summaries


def analyse_contradictions(data, selected_so_far):


    subtitles = []
    summaries = []

    for file_name in tqdm(data):
        if 'headings_inputs' not in data[file_name]:
            continue
        if 'summary_inputs' not in data[file_name]:
            continue
        for summary,subtitle in zip(data[file_name]['summary_inputs']\
                ['summary_healthline_entity_annotations'],\
                data[file_name]['headings_inputs']\
                ['headings_healthline_text'].keys()):
            if summary[0].strip() in selected_so_far or \
                   summary[0] in selected_so_far or \
                   is_insufficient(summary[0]):
                continue
            if any([x in summary[0].lower() for x in \
                    ['conflict','contradict','evidence is mixed']]):
                subtitles.append(subtitle)
                summaries.append(summary[0].strip())
    print(len(summaries))
    return summaries

def analyse_insufficient(data, selected_so_far):

    subtitles = []
    summaries = []

    for file_name in tqdm(data):
        if 'headings_inputs' not in data[file_name]:
            continue
        if 'summary_inputs' not in data[file_name]:
            continue
        for summary,subtitle in zip(data[file_name]['summary_inputs']\
                ['summary_healthline_entity_annotations'],\
                data[file_name]['headings_inputs']\
                ['headings_healthline_text'].keys()):
            if summary[0].strip() in selected_so_far or \
                    summary[0] in selected_so_far or \
                    is_contradiction(summary[0]):
                continue
            if any([x in summary[0].lower() for x in \
                    ['more research','further research',\
                    'additional research', 'more human research',\
                    'further human research', 'additional human research']]):
                subtitles.append(subtitle)
                summaries.append(summary[0].strip())
    print(len(summaries))
    return summaries

def certain_results(data, selected_so_far):

    subtitles = []
    summaries = []

    for file_name in tqdm(data):
        if 'headings_inputs' not in data[file_name]:
            continue
        if 'summary_inputs' not in data[file_name]:
            continue
        for summary, subtitle in zip(data[file_name]['summary_inputs']\
                ['summary_healthline_entity_annotations'],\
                data[file_name]['headings_inputs']\
                ['headings_healthline_text'].keys()):
            if summary[0].strip() in selected_so_far or \
                    summary[0] in selected_so_far:
                continue
            if not any([x in summary[0].lower() for x in \
                    ['more research','further research',\
                    'additional research','conflict','contradict','may']]):
                subtitles.append(subtitle)
                summaries.append(summary[0].strip())
    print(len(summaries))
    return summaries

def is_input_contradictions(data, input_pubmeds, file_name):

    pubmed_causes  = pickle.load(open("pubmed_causes.p","rb"))
    pubmed_causes_dicts = {}
    for pubmed, causes in pubmed_causes.items():
        pubmed_causes_dicts[pubmed] = {}
        for cause in causes:
            if cause[2] == 'None':
                continue
            pubmed_causes_dicts[pubmed][tuple(cause[:2])] = cause[2]


    all_causes = [pubmed_causes_dicts.get(pubmed,{}) \
            for pubmed in input_pubmeds]
    contradiction = False
    for I in range(len(all_causes)):                                                          
        for J in range(I+1,len(all_causes)):
            if I == J:                                                                              
                continue                                            
            for i_cause, i_value in all_causes[I].items():                                        
                for j_cause, j_value in all_causes[J].items():
                    if  i_value == 'None' or \
                        j_value == 'None':
                        continue                                     
                    if i_cause == j_cause and i_value != j_value:
                        contradiction = True
                        break                                                                     
                if contradiction:
                    break                                                                     
            if contradiction:
                break                                                                         
        if contradiction:
            break
    return contradiction 

def is_input_not_enough(data, input_pubmeds, file_name):

    if len(set(input_pubmeds)) == 1:
        return True
    return False

def is_input_scoping(data, input_pubmeds, file_name):

    relation_annotations = pickle.load(open("sentence_relation_annotations.p",\
            "rb"))
    populations = []
    for pubmed in input_pubmeds:
        if pubmed not in data['pubmed_sentences_annotations']:
            continue
        if 'pubmed_sentences_relation_annotations' not in \
                data['pubmed_sentences_annotations'][pubmed]:
            continue
        relations = data['pubmed_sentences_annotations'][pubmed]\
                ['pubmed_sentences_relation_annotations'][7]
        populations += [relation[0] for relation in relations]
    populations = list(set(populations))
    return len(populations) == 1

def is_input_certain(data, input_pubmeds, file_name):

    return not (is_input_contradictions(data, input_pubmeds, file_name) or\
            is_input_not_enough(data, input_pubmeds, file_name) or\
            is_input_scoping(data, input_pubmeds, file_name))


def input_contradictions(data, selected_so_far):

    subtitles = []
    summaries = []

    pubmed_causes  = pickle.load(open("pubmed_causes.p","rb"))

    pubmed_causes_dicts = {}
    for pubmed, causes in pubmed_causes.items():
        pubmed_causes_dicts[pubmed] = {}
        for cause in causes:
            if cause[2] == 'None':
                continue
            pubmed_causes_dicts[pubmed][tuple(cause[:2])] = cause[2]

    for file_name in tqdm(data):
        if 'summary_inputs' not in data[file_name]:
            continue
        if 'summary_pubmed_articles' not in data[file_name]['summary_inputs']:
            continue
        for summary,pubmeds in data[file_name]['summary_inputs']\
                ['summary_pubmed_articles'].items():
            if summary.strip() in selected_so_far:
                continue
            if len(pubmeds) == 0:
                continue
            all_causes = [pubmed_causes_dicts.get(pubmed,{})\
                    for pubmed in pubmeds]
            contradiction  = False
            for I in range(len(all_causes)):
                for J in range(I+1,len(all_causes)):
                    if I == J:
                        continue
                    for i_cause, i_value in all_causes[I].items():
                        for j_cause, j_value in all_causes[J].items():
                            if  i_value == 'None' or \
                                    j_value == 'None':
                                continue
                            if i_cause == j_cause and i_value != j_value:
                                if i_value in ['decreases','controls'] \
                                        and j_value in ['decreases','controls']:
                                    continue
                                if i_value in ['increases','satisfies'] \
                                        and j_value in ['increases','satisfies']:
                                    continue
                                contradiction = True
                                break
                            if contradiction:
                                break
                        if contradiction:
                            break
                    if contradiction:
                        break
                if contradiction:
                    break
            if contradiction:
                summaries.append(summary.strip())
    print(len(summaries))
    return summaries


def input_not_enough(data, selected_so_far):

    summaries = []
    subtitles = []
    
    for file_name in tqdm(data):
        if 'summary_inputs' not in data[file_name]:
            continue
        if 'summary_pubmed_articles' not in data[file_name]['summary_inputs']:
            continue
        for summary,pubmeds in data[file_name]['summary_inputs']\
                ['summary_pubmed_articles'].items():
            if summary.strip() in selected_so_far:
                continue
            if len(pubmeds) == 0:
                continue
            if len(set(pubmeds)) in [1]:
                summaries.append(summary.strip())
    print(len(summaries))
    return summaries

def input_scoping(data, selected_so_far):

    summaries = []
    subtitles = []

    relation_annotations = pickle.load(open("sentence_relation_annotations.p",\
            "rb"))

    sentence_populations = {}

    for file_name in tqdm(data):
        if 'summary_inputs' not in data[file_name]:
            continue
        if 'summary_pubmed_articles' not in data[file_name]['summary_inputs']:
            continue
        for summary,pubmeds in data[file_name]['summary_inputs']\
                ['summary_pubmed_articles'].items():
            if summary.strip() in selected_so_far:
                continue
            if len(pubmeds) == 0:
                continue
            if summary.strip() not in relation_annotations:
                continue
            #populations = \
            #        [relation_annotations[summary.strip()][1][7][i][0]\
            #        for i in range(len(\
            #        relation_annotations[summary.strip()][1][7]))]
            populations = []
            for p in data['pubmed_sentences_annotations']:
                if p not in pubmeds:
                    continue
                current_population = []
                if 'pubmed_sentences_entity_annotations' not in \
                        data['pubmed_sentences_annotations'][p]:
                    continue
                for sent_ann in data['pubmed_sentences_annotations'][p]\
                        ['pubmed_sentences_entity_annotations']:
                    sent,ann = sent_ann[0], sent_ann[1]
                    current_population += _get_entities(sent.split(),\
                            ann.split(),'Population')
                populations += current_population
            if len(populations) == 0:
                continue
            if any([x in populations for x in ['men','women',\
                    'participants','human','humans','subjects']]):
                continue
            #if any([[x in y for y in populations] for x in ['healthy']]):
            #    continue
            if len(set(populations)) <= 1:
                summaries.append(summary.strip())
                sentence_populations[summary.strip()] = \
                        list(set(populations))[0]
    print(len(summaries))
    pickle.dump(sentence_populations, open("scoping_populations.p","wb"))
    return summaries
   
def input_certain(data, selected_so_far):

    summaries = []

    for file_name in tqdm(data):
        if 'summary_inputs' not in data[file_name]:
            continue
        if 'summary_pubmed_articles' not in data[file_name]['summary_inputs']:
            continue
        for summary,pubmeds in data[file_name]['summary_inputs']\
                ['summary_pubmed_articles'].items():
            if len(pubmeds) > 0 and summary.strip() not in selected_so_far:
                summaries.append(summary.strip())
    print(len(summaries))
    return summaries


def get_sentence_annotations():

    dictionary = pickle.load(open("categorized_output_summaries.p","rb"))

    f = open("categorized_entity_input.txt","w")

    for type,sentences in dictionary.items():
        for sentence in sentences:
            words = nltk.tokenize.word_tokenize(sentence)
            for word in words:
                f.write(word + " O\n")
            f.write("\n")
    f.close()

def is_contradiction(sentence):
    if any([x in sentence.lower() for x in \
            ['conflict','contradict','evidence is mixed','mixed results']]):
        return True
    return False

def is_insufficient(sentence):
    if any([x in sentence.lower() for x in \
            ['more research','further research',\
            'additional research', 'more human research',\
            'further human research', 'additional human research',\
            'more high - quality studies']]):
        return True
    return False



def check_outputs(input_file_name,output_file_name,incorrect_input,\
        incorrect_output):

    output_lines = open(output_file_name, "r").readlines()
    input_lines  = open(input_file_name, "r").readlines()
    incorrect_inputs = []
    incorrect_outputs = []
    for output, input in zip(output_lines, input_lines):
        contradiction = is_contradiction(output.strip())
        insufficient  = is_insufficient(output.strip())
        if contradiction and input.startswith("#4"):
            continue
        if insufficient and input.startswith("#2"):
            continue
        if (not (contradiction or insufficient)) and (input.startswith("#1") \
                or input.startswith("#3")):
            continue
        incorrect_inputs.append(input)
        incorrect_outputs.append(output)
    print(len(incorrect_inputs))
    f_in = open(incorrect_input, "w")
    f_out= open(incorrect_output,"w")
    for input,output in zip(incorrect_inputs,incorrect_outputs):
        f_in.write(input.strip()+"\n")
        f_out.write(output.strip()+"\n")
    f_in.close()
    f_out.close()


def check_pubmed_outputs(data):

    pubmed_causes = pickle.load(open("pubmed_causes.p","rb"))

    consider_sentences = set()

    for pubmed in tqdm(data['pubmed_sentences_annotations']):
        if 'pubmed_sentences_entity_annotations' not in data['pubmed_sentences_annotations'][pubmed]:
            continue
        for sentence_tuple in data['pubmed_sentences_annotations']\
                [pubmed]['pubmed_sentences_entity_annotations']:
            sentence, _ = sentence_tuple
            current_causes = pubmed_causes.get(pubmed,set())
            cause_consider = set([tuple(instances[:2]) for instances\
                    in current_causes if instances[2]!='None'])
            if any([cause[0] in sentence and cause[1] in sentence\
                    for cause in cause_consider]):
                consider_sentences.add(sentence)
    categorized_pubmed_outputs = {}
    for sentence in consider_sentences:
        if is_contradiction(sentence):
            categorized_pubmed_outputs['contradiction'] = \
                    categorized_pubmed_outputs.setdefault('contradiction',[])+\
                    [sentence]
        elif is_insufficient(sentence):
            categorized_pubmed_outputs['not_enough'] = \
                    categorized_pubmed_outputs.setdefault('not_enough',[])+\
                    [sentence]
        else:
            categorized_pubmed_outputs['certain'] = \
                    categorized_pubmed_outputs.setdefault('certain',[])+\
                    [sentence]
    pickle.dump(categorized_pubmed_outputs,open("categorized_pubmed_"+\
        "summaries.p","wb"))

def gather_multi_sentence_summaries(data):

    sentences_pubmeds = {}
    sentence_string_pubmeds = {}
    grouped_sentences = []
    categorized_sentences = {}
    sentence_categories = {}
    category_counts = {}
    sentence_sentence_splits = {}
    sentence_file_names = {}
    for file_name in data:
        if 'headings_inputs' not in data[file_name]:#['headings_inputs']['headings_healthline_text']
            continue
        if 'distant_mapped_sentences' not in data[file_name]:
            continue
        for heading in data[file_name]['headings_inputs']\
                ['headings_healthline_text']:
            sentences = data[file_name]['headings_inputs']\
                    ['headings_healthline_text'][heading]
            if len([x for x in sentences if x!='' and x in\
                    data[file_name]['distant_mapped_sentences']]) == 0:
                continue
            sentences = [x \
                    for x in sentences if x!='' and x in\
                    data[file_name]['distant_mapped_sentences']]
            pubmeds   = [data[file_name]['distant_mapped_sentences'][x]\
                    for x in sentences]
            grouped_sentences.append(" ".join(sentences).strip())
            sentences_pubmeds[tuple(sentences)] = pubmeds
            sentence_string_pubmeds[" ".join(sentences).strip()] = pubmeds
            sentence_file_names[tuple(sentences)] = file_name
            sentence_sentence_splits[" ".join(sentences).strip()] = \
                    sentences
            sentence_category = "certain"
            if is_contradiction(" ".join(sentences).strip()):
                sentence_category = "contradiction"
            elif is_insufficient(" ".join(sentences).strip()):
                sentence_category = "not_enough"
            sentence_categories[" ".join(sentences).strip()] = sentence_category
            category_counts[sentence_category] = category_counts.setdefault(\
                    sentence_category,0) + 1
            categorized_sentences[sentence_category] = categorized_sentences.\
                    setdefault(sentence_category,[]) + \
                    [" ".join(sentences).strip()]
    old_sentences_pubmeds = pickle.load(open("multi_sentences_pubmeds.p","rb"))
    pickle.dump(categorized_sentences,open(\
            "output_categorized_multi_sentences.p","wb"))
    pickle.dump(sentences_pubmeds,open(\
            "multi_sentences_pubmeds.p","wb"))
    assert sentences_pubmeds == old_sentences_pubmeds
    pickle.dump(sentence_file_names,open(\
        "multi_sentence_file_names.p","wb"))
    create_entity_annotations_file(sum(sentence_sentence_splits.values(),[]),\
            "/data/rsg/nlp/darsh/aggregator/crawl_websites/"+\
            "NUT/model_sentences.txt")
    _perform_tagging("demo.model_evaluation_decode.config")
    machine_r_sentences, machine_labels = _read_entity_file(\
            "/data/rsg/nlp/darsh/aggregator/crawl_websites"\
            "/NUT/model_sentences_out.txt")
    machine_sentence_entities = get_sentence_entities(machine_r_sentences,\
            machine_labels)
    machine_causes_input, machine_contains_input = \
            create_relation_annotations_file(machine_sentence_entities,"model")
    machine_causes_output    = produce_re_outputs("model_causes.jsonl",\
            "t3_causes")
    machine_contains_output  = produce_re_outputs("model_contains.jsonl",\
            "t3_contains")
    machine_causes           = get_predicted_relations(\
            machine_causes_input,machine_causes_output)
    machine_contains         = get_predicted_relations(\
            machine_contains_input,machine_contains_output)
    number_of_good_sentences = 0
    good_category_counts= {}
    food_terms      = sum([machine_sentence_entities[x].get('Food',[]) for x in \
            machine_sentence_entities.keys()],[])
    nutrition_terms = sum([machine_sentence_entities[x].get('Nutrition',[]) for x in \
            machine_sentence_entities.keys()],[])
    condition_terms = sum([machine_sentence_entities[x].get('Condition',[]) for x in\
            machine_sentence_entities.keys()],[])
    embeddings      = read_embeddings()
    food_closest    = get_closest_terms(food_terms,embeddings)
    nutr_closest    = get_closest_terms(nutrition_terms,embeddings)
    cond_closest    = get_closest_terms(condition_terms,embeddings)

    train_source    = open("multi_train.source","w")
    train_target    = open("multi_train.target","w")
    multiplier      = {'certain':0,'not_enough':14,'contradiction':44}
    data_point_counts={}

    sentence_tuple_causes = {}
    for sentence_tuples in sentence_file_names:
        sentence_tuple_causes[sentence_tuples] = []
        for sentence in sentence_tuples:
            sentence_tuple_causes[sentence_tuples].append([\
                    machine_causes.get(sentence,[]),\
                    machine_contains.get(sentence,[])])

    pickle.dump(sentence_tuple_causes,\
            open("sentence_tuple_causes.p","wb"))
    for grouped_sentence in grouped_sentences:
        split_sentences = sentence_sentence_splits[grouped_sentence]
        good_candidate  = False
        for sentence in split_sentences:
            sentence_cause = machine_causes.get(sentence,[])
            sentence_contain = machine_contains.get(sentence,[])
            if not(len(sentence_cause) == 0 and len(sentence_contain) == 0):
                good_candidate = True
                break
        if good_candidate:
            good_category_counts[sentence_categories[grouped_sentence]] = \
                    good_category_counts.setdefault(\
                    sentence_categories[grouped_sentence],0) + 1
            number_of_good_sentences += 1
        else:
            continue
        net_input = []
        net_output= []
        food_names=[]
        cond_names=[]
        nutr_names=[]
        for sentence in split_sentences:
            food_names += machine_sentence_entities[sentence].get('Food',[])
            cond_names += machine_sentence_entities[sentence].get('Condition',[])
            nutr_names += machine_sentence_entities[sentence].get('Nutrition',[])
            sentence_causes = machine_causes.get(sentence,[])
            sentence_contains = machine_contains.get(sentence,[])
            indices = [sentence.index(c[1]) for c in sentence_contains]
            indices = np.argsort(indices)
            sentence_contains = [sentence_contains[i] for i in indices]
            indices = [sentence.index(c[1]) for c in sentence_causes]
            indices = np.argsort(indices)
            sentence_causes   = [sentence_causes[i] for i in indices]
            if len(sentence_causes) == 0 and len(sentence_contains) == 0:
                continue
            good_text = "<blank> "
            good_text += (sentence_contains+sentence_causes)[0][0] + " "
            for c in sentence_contains:
                good_text += "<contains> " + c[1] + " "
            for c in sentence_causes:
                good_text += "<"+c[2]+"> " + c[1] + " "
            good_text += "<blank>"
            net_input.append(good_text)
            net_output.append(sentence)
        if len(net_input) == 0:
            continue
        train_input = " |SEP| ".join(net_input).strip()
        train_output= " |SEP| ".join(net_output).strip()
        sample_text = random.choice(categorized_sentences[\
                sentence_categories[grouped_sentence]])
        if len(sample_text.split())+len(train_input.split())+len(train_output.split()) > 220:
            continue
        data_point_counts[sentence_categories[grouped_sentence]] =\
                data_point_counts.setdefault(sentence_categories[grouped_sentence],0) + 1
        train_source.write(sample_text + " |SEN| " + train_input + "\n")
        train_target.write(train_output + "\n")
        for entity_terms,entity_closest in zip([food_names,nutr_names,cond_names],\
                [food_closest,nutr_closest,cond_closest]):
            for entity_term in entity_terms:
                entity_closests = entity_closest[entity_term][:\
                        multiplier[sentence_categories[grouped_sentence]]]
                for entity_clos in entity_closests:
                    sample_text = random.choice(categorized_sentences[\
                            sentence_categories[grouped_sentence]])
                    n_train_input = train_input.replace(entity_term,entity_clos)
                    n_train_output= train_output.replace(entity_term,entity_clos)
                    if len(sample_text.split())+len(n_train_input.split())+len(n_train_output.split()) > 220:
                        continue
                    train_source.write(sample_text + " |SEP| " +\
                            n_train_input + "\n")
                    train_target.write(n_train_output + "\n")
                    data_point_counts[sentence_categories[grouped_sentence]] =\
                            data_point_counts.setdefault(sentence_categories[grouped_sentence],0) + 1
    train_source.close()
    train_target.close()

    print(number_of_good_sentences)
    print(category_counts)
    print(good_category_counts)
    print(data_point_counts)

    pubmed_causes = pickle.load(open("pubmed_causes.p","rb"))
    test_cat_counts = {}

    cluster_extractive_multi_summaries = jsonlines.open(\
            "cluster_extractive_multi_summaries.jsonl","r")
    input_cluster_text = open("multi_test.source","w")
    pubmed_multi_test  = open("multi_test.pubmed","w")
    metadata           = json.load(open("annotated_metadata5.json","r"))
    for data_point in cluster_extractive_multi_summaries:
        file_name         = data_point['file_name']
        pubmed_sentences  = metadata[file_name]['pubmed_sentences']
        pubmed_titles     = {}
        pubmed_title_representations = {}
        cluster_sentences = data_point['clustered_sentences']
        cluster_numbers   = data_point['cluster_numbers']
        current_pubmeds   = list(set(sum(sentences_pubmeds\
                [tuple(data_point['gold'][0])],[])))
        ignore_instance   = False
        for cluster_sentence in cluster_sentences:
            if cluster_sentence[0][-3] not in current_pubmeds:
                ignore_instance = True
                break
        if ignore_instance:
            continue
        current_causes    = []
        for pubmed in current_pubmeds:
            for p_c in pubmed_causes.get(pubmed,[]):
                current_causes.append(p_c + [pubmed])
        for pubmed in current_pubmeds:
            pubmed_titles[pubmed] = " ".join(pubmed_sentences[pubmed]\
                    [0][0]).strip()
            pubmed_title_representations[pubmed] = _get_entity_embeddings(\
                    pubmed_titles[pubmed],embeddings)
        is_cont  = False
        for i,current_cause1 in enumerate(current_causes):
            for j,current_cause2 in enumerate(current_causes):
                if current_cause1[-1] == current_cause2[-1]:
                    continue
                if current_cause1[:2] == current_cause2[:2] and \
                        current_cause1[2] != current_cause2[2]:
                    is_cont = True
                    cluster_sentences = [[current_cause1,"",""],\
                            [current_cause2,"",""]]
                    break
            if is_cont:
                break
        food_name         = data_point['food_name']
        category_name     = 'certain'
        if is_cont:
            category_name = 'contradiction'
        candidate_titles  = []
        for cause in cluster_sentences:
            candidate_titles.append(pubmed_title_representations.get(\
                    cause[0][-3],np.array([0]*50)))
        cluster_pubmeds = {}
        for pubmed in current_pubmeds:
            pubmed_repr = pubmed_title_representations.get(\
                    pubmed,np.array([0]*50))
            dot_products= [np.dot(pubmed_repr,p) for p in candidate_titles]
            best_index  = dot_products.index(max(dot_products))
            cluster_pubmeds[best_index] = cluster_pubmeds.setdefault(best_index,[]) + [pubmed]
        clusters_pubmeds_list = []
        for ind in range(len(cluster_sentences)):
            clusters_pubmeds_list.append(cluster_pubmeds.get(ind,[]))
        pubmed_multi_test.write(str(clusters_pubmeds_list) + "\n")
        sample_text = random.choice(categorized_sentences[category_name])
        test_cat_counts[category_name] = test_cat_counts.setdefault(\
                category_name,0) + 1
        net_input   = []
        for cluster_sentence in cluster_sentences:
            sent = cluster_sentence[0]
            if food_name != "":
                sent[0] = food_name
            curr_input = "<blank> " + sent[0] + " <" + sent[2].split()[0] + "> " + sent[1] + " <blank>"
            net_input.append(curr_input)
        input_text = " |SEN| ".join(net_input).strip()
        input_cluster_text.write(sample_text + " |SEP| " + input_text + "\n")
    input_cluster_text.close()
    pubmed_multi_test.close()
    print(test_cat_counts)


if __name__ == "__main__":

    data = json.load(open("annotated_metadata5.json","r"))
    #gather_multi_sentence_summaries(data)
    #import sys; sys.exit()
    #a_summaries = analyse_insufficient(data,[])
    #output_scoping_summaries = analyse_scoping(data,\
    #        set(a_summaries))
    #c_summaries = analyse_contradictions(data, \
    #        set(output_scoping_summaries+a_summaries))
    #certain_summaries = certain_results(data, set(output_scoping_summaries\
    #        +c_summaries+a_summaries))
    #dictionary_outputs= {'contradiction':c_summaries,'not_enough':a_summaries,\
    #        'certain':certain_summaries,'scoping':output_scoping_summaries}
    #pickle.dump(dictionary_outputs,open("categorized_output_summaries.p","wb"))
    #input_contradictions(data)
    ne_summaries = input_not_enough(data,[])
    scoping_summaries = input_scoping(data,set(ne_summaries))
    contradiction_summaries = input_contradictions(data,\
            set(ne_summaries+scoping_summaries))
    certain_summaries = input_certain(data, set(ne_summaries+\
            scoping_summaries+contradiction_summaries))
    print(len(ne_summaries),len(scoping_summaries),\
            len(contradiction_summaries),len(certain_summaries))
    print(len(set(ne_summaries+scoping_summaries+contradiction_summaries)))
    dict = {'not_enough':ne_summaries, 'scoping':scoping_summaries,\
            'contradiction':contradiction_summaries, \
            'certain':certain_summaries}
    pickle.dump(dict,open("new_categorized_input_summaries.p","wb"))
    import sys; sys.exit()
    get_sentence_annotations()
    check_outputs("train_expanded.source",
            "/data/rsg/nlp/darsh/MatrixEmbedding/fairseq/infilling-categorized-bin/train.hypo",
            "train.source8",
            "train.target8")
    check_pubmed_outputs(data)
