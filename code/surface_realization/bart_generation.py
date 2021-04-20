import torch
from fairseq.models.bart import BARTModel
import regex as re
from tqdm import tqdm
import pdb
import jsonlines


bart = BARTModel.from_pretrained(
        './infilling_checkpoint/',
        checkpoint_file='checkpoint_last.pt',
        data_name_or_path='infilling-categorized-bin'
    )

bart.cuda()
bart.eval()
bart.half()

count = 1
total_data = 160
beam = 5
bsz = total_data//beam

lines = open("/data/rsg/nlp/darsh/aggregator/"
        "crawl_websites/NUT/test_meteor.source","r").readlines()

output_writer = jsonlines.open("infilling-categorized-bin/test_meteor.jsonl","w")

with open("/data/rsg/nlp/darsh/aggregator/"\
        +"crawl_websites/NUT/test_meteor.source") as source,\
        open('infilling-categorized-bin/test_meteor.hypo', 'w') as fout,\
        open('infilling-categorized-bin/train.surface','w') as fout2:
    sline = source.readline().strip()
    slines = []
    gold_lines = open("/data/rsg/nlp/darsh/aggregator/"\
            "crawl_websites/NUT/test.gold","r").readlines()
    g_lines = []
    for sline,gold_line in tqdm(zip(lines,gold_lines)):
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=beam)
            for hypothesis,sline,g_line in zip(hypotheses_batch,slines,g_lines):
                fout.write(hypothesis + '\n')
                dict = {'gold':g_line,'output':hypothesis}
                output_writer.write(dict)
                fout.flush()
                begins = [m.start() for m in re.finditer('<', sline)]
                endings= [m.start() for m in re.finditer('>', sline)]
                substrings = [m.start() for m in re.finditer('#', hypothesis)]\
                        + [len(hypothesis)]
                input_replace = [sline[b:e+1] for b,e in zip(begins,endings)]
                output_replace= [hypothesis[substrings[i]+1:\
                        substrings[i+1]].strip() for i in \
                        range(len(substrings)-1)]
                output_text   = sline
                for input,output in zip(input_replace, output_replace):
                    output_text = output_text.replace(input,output,1)
                fout2.write(output_text + '\n')
                fout2.flush()
            slines = []
            g_lines= []
        slines.append(sline.strip())
        g_lines.append(gold_line.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=beam)
        for hypothesis,sline,g_line in tqdm(zip(hypotheses_batch,slines,g_lines)):
            fout.write(hypothesis + '\n')
            dict = {'gold':g_line,'output':hypothesis}
            output_writer.write(dict)
            fout.flush()
            begins = [m.start() for m in re.finditer('<', sline)]
            endings= [m.start() for m in re.finditer('>', sline)]
            substrings = [m.start() for m in re.finditer('#', hypothesis)]\
                    + [len(hypothesis)]
            input_replace = [sline[b:e+1] for b,e in zip(begins,endings)]
            output_replace= [hypothesis[substrings[i]+1:substrings[i+1]].strip()\
                    for i in range(len(substrings)-1)]
            output_text   = sline
            for input,output in zip(input_replace, output_replace):
                output_text = output_text.replace(input,output,1)
            fout2.write(output_text + '\n')
            fout2.flush()
print(count)
output_writer.close()
