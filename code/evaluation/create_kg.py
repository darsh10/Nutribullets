import jsonlines
import sys
import os
from shutil import copyfile
import argparse
import pdb
import pickle
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

sys.path.append('/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/')
from gather_annotations import _perform_tagging, _read_entity_file
from get_baseline_performances import _get_entities, read_embeddings,\
        _compare_causes, _compare_contains


def get_phrase_embeddings(string, embeddings):

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


def produce_re_outputs(input_file_path,\
        model="t3_causes"):
    
    copyfile(input_file_path,"T4_causes.jsonl")
    os.chdir("/data/rsg/nlp/darsh/pytorch-pretrained-BERT")
    os.system("python examples/run_re.py --task_name "\
            "re_task --do_eval --do_lower_case --data_dir /data/rsg/nlp/darsh"\
            "/aggregator/crawl_websites/NUT/ --bert_model bert-base-uncased "\
            "--max_seq_length 128 --train_batch_size 32 --learning_rate 5e-5"\
            " --num_train_epochs 3.0 --output_dir "\
            + model + " --eval_batch_size 32 --output_preds")
    os.chdir("/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT")
    results_file = jsonlines.open("/data/rsg/nlp/darsh/"\
            "pytorch-pretrained-BERT/"+model+"/preds.jsonl","r")
    results = []
    for r in results_file:
        results.append(r)
    return results

def produce_causes(input_file_path, model="T4_redo_causes2"):
    copyfile(input_file_path,"T4_dev_causes.jsonl")
    os.chdir("/data/rsg/nlp/darsh/pytorch-pretrained-BERT")
    os.system("python examples/run_causes.py --task_name "\
            "re_task --do_eval --do_lower_case --data_dir /data/rsg/nlp/darsh"\
            "/aggregator/crawl_websites/NUT/ --bert_model bert-base-uncased "\
            "--max_seq_length 128 --train_batch_size 32 --learning_rate 5e-5"\
            " --num_train_epochs 3.0 --output_dir "\
            + model + " --eval_batch_size 32 --output_preds")
    os.chdir("/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT")
    results_file = jsonlines.open("/data/rsg/nlp/darsh/"\
            "pytorch-pretrained-BERT/"+model+"/preds.jsonl","r")
    results = []
    for r in results_file:
        results.append(r)
    return results


def create_entity_annotations_file_ind(sentences, index, file_path):

    output_file = open(file_path, "w")
    
    for sentence in sent_tokenize(sentences[index]):
        words   = sentence.split()
        for word in words:
            output_file.write(word + " O\n")
        output_file.write("\n")
    output_file.close()

def create_entity_annotations_file(sentences, file_path):

    output_file = open(file_path, "w")
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            output_file.write(word + " O\n")
        output_file.write("\n")
    output_file.close()


def get_sentence_entities(sentences, labels):

    sentence_entities = {}
    for sentence,label_str in zip(sentences,labels):
        sentence_entities[sentence] = {}
        for entity_type in ['Food', 'Condition', 'Nutrition']:
            sentence_entities[sentence][entity_type] \
                = _get_entities(sentence.split(),label_str.split(),entity_type)
    return sentence_entities


def create_relation_annotations_file(sentence_entities, file_substr):

    f_causes = jsonlines.open(file_substr + "_causes.jsonl","w")
    f_contains = jsonlines.open(file_substr + "_contains.jsonl","w")
    contains_dictionary = []
    causes_dictionary   = []
    for sentence in tqdm(sentence_entities):
        considered = set()
        for x in sentence_entities[sentence]['Food']\
                 + sentence_entities[sentence]['Nutrition']:
            for y in sentence_entities[sentence]['Nutrition']:
                if tuple([x,y]) in considered:
                    continue
                considered.add(tuple([x,y]))
                if x == y:
                    continue
                if x not in sentence or y not in sentence:
                    continue
                assert x in sentence
                assert y in sentence
                modified_text = sentence.replace(x, ' <X> ' + x + ' </X> ')
                modified_text = modified_text.replace(y, ' <Y> ' + y + ' </Y> ')
                if x not in modified_text or y not in modified_text:
                    continue
                x_index = modified_text.index(x)
                y_index = modified_text.index(y)
                min_index= min(x_index,y_index)
                max_index= max(x_index,y_index)
                #if modified_text[min_index:max_index].count(" ") > 45:
                    #print(modified_text[min_index:max_index].count(" "))
                #    continue
                #modified_text = modified_text[min_index-30:max_index+30]
                dict = {'sentence':modified_text, 'gold_label':'None', 'uid':0,\
                        'original_sentence':sentence}
                contains_dictionary.append(dict)
                f_contains.write(dict)
        for x in sentence_entities[sentence]['Food']\
                + sentence_entities[sentence]['Nutrition']\
                + sentence_entities[sentence]['Condition']:
            for y in sentence_entities[sentence]['Condition']:
                if tuple([x,y]) in considered:
                    continue
                considered.add(tuple([x,y]))
                if x == y:
                    continue
                if x not in sentence or y not in sentence:
                    continue
                modified_text = sentence.replace(x, ' <X> ' + x + ' </X> ')
                modified_text = modified_text.replace(y, ' <Y> ' + y + ' </Y> ')
                if x not in modified_text or y not in modified_text:
                    continue
                x_index = modified_text.index(x)
                y_index = modified_text.index(y)
                #if modified_text[min_index:max_index].count(" ") > 45:
                    #print(modified_text[min_index:max_index].count(" "))
                #    continue
                #min_index= min(x_index,y_index)
                #max_index= max(x_index,y_index)
                #if modified_text[min_index:max_index].count(" ") > 45:
                #    continue
                #modified_text = modified_text[min_index-30:max_index+30]
                dict = {'sentence':modified_text, 'gold_label':'None', 'uid':0,\
                        'original_sentence':sentence}
                causes_dictionary.append(dict)
                f_causes.write(dict)
    f_contains.close()
    f_causes.close()
    return causes_dictionary, contains_dictionary


def get_predicted_relations(input_list, output_list):

    sentence_relations = {}
    for input,output in zip(input_list,output_list):
        original_sentence = input['original_sentence']
        relation_sentence = input['sentence']
        x_string          = relation_sentence[relation_sentence.find("<X>")+3:\
                relation_sentence.find("</X>")].strip()
        y_string          = relation_sentence[relation_sentence.find("<Y>")+3:\
                relation_sentence.find("</Y>")].strip()
        pred_label        = output['pred_label']
        if pred_label != 'None':
            sentence_relations[original_sentence] = \
                    sentence_relations.setdefault(original_sentence,[]) +\
                    [[x_string,y_string,pred_label]]
    return sentence_relations


def evaluate_relations(gold_sentences, machine_sentences, gold_all_rels,\
        machine_all_rels, embeddings):

    recalls = []
    precisions = []
    f1s     = []
    missing_list = []
    for gold,machine in zip(gold_sentences,machine_sentences):
        missing_dict = {}
        missing_dict['sentence'] = gold
        missing_dict['output']   = machine
        missing_dict['missing_rels'] = []
        missing_dict['matching_rels']= []
        gold_relations = gold_all_rels.get(gold,[])
        machine_relations = machine_all_rels.get(machine,[])
        matching_relations = [_compare_causes(g_r,machine_relations,embeddings,"")\
                for g_r in gold_relations]
        num_matching   = sum(matching_relations)
        considered_rels= set()
        for matched,g_r in zip(matching_relations,gold_relations):
            if tuple(g_r[:2]) in considered_rels:
                continue
            else:
                considered_rels.add(tuple(g_r[:2]))
            if not matched:
                missing_dict['missing_rels'].append(g_r)
            else:
                missing_dict['matching_rels'].append(g_r)
        if len(gold_relations) > 0:
            recalls.append(num_matching/len(gold_relations))
            #recalls.append(num_matching/len(considered_rels))
        else:
            continue
        if len(machine_relations) > 0:
            precisions.append(num_matching/len(machine_relations))
        else:
            assert len(gold_relations) > 0
            precisions.append(0)
        den = 1 if precisions[-1]+recalls[-1] == 0 else\
                 precisions[-1] + recalls[-1]
        num = 2 * precisions[-1] * recalls[-1]
        f1s.append(num/den)
        missing_list.append(missing_dict)
    return recalls, precisions, f1s, missing_list
        
def evaluate_entities(all_output_entities, all_target_entities, embeddings):

    accuracies = []
    for output_entities,target_entities in zip(all_output_entities,\
            all_target_entities):
        new_output_entities = []
        for output_entity in output_entities:
            if output_entity.strip()[-1].isalnum():
                new_output_entities.append(output_entity.strip())
            else:
                new_output_entities.append(output_entity.strip()[:-1])
        output_entities = new_output_entities

        if len(output_entities) == 0 or len(target_entities) == 0:
            accuracies.append(0)
            continue
        output_representations = np.stack([get_phrase_embeddings(output_entity,\
                embeddings) for output_entity in output_entities],axis=0)
        target_representations = np.stack([get_phrase_embeddings(target_entity,\
                embeddings) for target_entity in target_entities],axis=0)
        max_products           = np.max(np.dot(output_representations,\
                target_representations.transpose()),axis=1)
        #for output_entity,max_product in zip(output_entities,max_products):
            #if max_product <= 0.7:
            #    pdb.set_trace()
        accuracies.append(sum([x>0.7 for x in max_products])/\
                len(output_entities))
    return accuracies

def compare_results(args, analysis_file):

    file_names     = []
    gold_sentences = []
    machine_sentences = []
    if args.file_path is not None:
        input_reader = jsonlines.open(args.file_path, "r")
        input_instances = []
        gold_string = 'gold'
        output_string='output'
        for r in input_reader:
            gold_sentences.append(r[gold_string].replace('\n',' '))
            machine_sentences.append(r[output_string].replace('\n',' '))
            input_instances.append(r)
    else:
        gold_lines = open(args.gold, "r").readlines()
        output_lines = open(args.output, "r").readlines()
        for out,gold in zip(output_lines,gold_lines):
            machine_sentences.append(out.strip())
            gold_sentences.append(gold.strip())

    machine_r_sentences = []
    machine_labels      = []
    gold_r_sentences    = []
    gold_labels         = []
    #for i in range(len(input_instances)):

    #    create_entity_annotations_file_ind(machine_sentences, i,\
    #        "/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/model_sentences.txt")
    #    create_entity_annotations_file_ind(gold_sentences, i,\
    #        "/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/gold_sentences.txt")

    #    _perform_tagging("demo.model_evaluation_decode.config")
    #    _perform_tagging("demo.gold_evaluation_decode.config")

    #    machine_r_sentence, machine_label = _read_entity_file(\
    #        "/data/rsg/nlp/darsh/aggregator/crawl_websites"\
    #        "/NUT/model_sentences_out.txt")
    #    gold_r_sentence, gold_label       = _read_entity_file(\
    #        "/data/rsg/nlp/darsh/aggregator/crawl_websites"\
    #        "/NUT/gold_sentences_out.txt")
    #    machine_r_sentences.append(" ".join(machine_r_sentence).strip())
    #    machine_labels.append(" ".join(machine_label).strip())
    #    gold_r_sentences.append(" ".join(gold_r_sentence).strip())
    #    gold_labels.append(" ".join(gold_label).strip())
    create_entity_annotations_file(machine_sentences,\
            "/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/model_sentences.txt")
    create_entity_annotations_file(gold_sentences,\
            "/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT/gold_sentences.txt")
    _perform_tagging("demo.model_evaluation_decode.config")
    _perform_tagging("demo.gold_evaluation_decode.config")
    machine_r_sentences, machine_labels = _read_entity_file(\
            "/data/rsg/nlp/darsh/aggregator/crawl_websites"\
            "/NUT/model_sentences_out.txt")
    gold_r_sentences, gold_labels = _read_entity_file(\
            "/data/rsg/nlp/darsh/aggregator/crawl_websites"\
            "/NUT/gold_sentences_out.txt")


    machine_sentence_entities = get_sentence_entities(machine_r_sentences, \
            machine_labels)
    machine_nutrition_entities= []
    machine_condition_entities= []
    file_name_m_conditions    = {}
    file_name_m_nutritions    = {}
    file_name_condition_counts= {}
    file_name_nutrition_counts= {}


    for m_s, m_l, f_n in zip(machine_r_sentences, machine_labels, file_names):
        n_e = _get_entities(m_s.split(),m_l.split(),'Nutrition')
        c_e = _get_entities(m_s.split(),m_l.split(),'Condition')
        file_name_m_conditions[f_n] = file_name_m_conditions.setdefault(f_n,[]\
                ) + c_e
        file_name_m_nutritions[f_n] = file_name_m_nutritions.setdefault(f_n,[]\
                ) + n_e
        file_name_condition_counts[f_n] = len(set(file_name_m_conditions[f_n]))
        file_name_nutrition_counts[f_n] = len(set(file_name_m_nutritions[f_n]))
        machine_nutrition_entities.append(n_e)
        machine_condition_entities.append(c_e)

    gold_sentence_entities    = get_sentence_entities(gold_r_sentences,\
            gold_labels)

    
    machine_causes_input, machine_contains_input = \
            create_relation_annotations_file(machine_sentence_entities,"model")
    gold_causes_input, gold_contains_input = \
            create_relation_annotations_file(gold_sentence_entities,"gold")
    
    machine_causes_output    = produce_re_outputs("model_causes.jsonl",\
                                                    "t3_causes")
    machine_contains_output  = produce_re_outputs("model_contains.jsonl",\
                                                    "t3_contains")
    gold_causes_output       = produce_re_outputs("gold_causes.jsonl",\
                                                    "t3_causes")
    gold_contains_output     = produce_re_outputs("gold_contains.jsonl",\
                                                    "t3_contains")


    machine_causes           = get_predicted_relations(machine_causes_input,\
                                                    machine_causes_output)
    machine_contains         = get_predicted_relations(machine_contains_input,\
                                                    machine_contains_output)
    gold_causes              = get_predicted_relations(gold_causes_input,\
                                                    gold_causes_output)
    gold_contains            = get_predicted_relations(gold_contains_input,\
                                                    gold_contains_output)

    embeddings               = read_embeddings()

    recalls, precision, f1s, missing_list  = \
            evaluate_relations(gold_sentences, machine_sentences,\
            gold_causes, machine_causes, embeddings)
    
    analysis_file            = jsonlines.open(args.analysis_output,'w')
    for missing_dict in missing_list:
        analysis_file.write(missing_dict)
    if len(recalls) > 0:
        print(sum(recalls)/len(recalls))
    else:
        print(0)
    recalls, precision, f1s, missing_list = \
            evaluate_relations(machine_sentences, gold_sentences,\
            machine_causes, gold_causes, embeddings)
    if len(recalls) > 0:
        print(sum(recalls)/len(recalls))
    else:
        print(0)
    analysis_file.close()

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Run sentiment evaluation')
    parser.add_argument('--file_path',default=None)
    parser.add_argument('--output',default='a.txt')
    parser.add_argument('--gold',default='b.txt')
    parser.add_argument('--analysis_output',default='kg_analysis.jsonl')

    args = parser.parse_args()

    compare_results(args, args.analysis_output)
