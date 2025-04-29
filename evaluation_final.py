import os, sys, json
sys.path.append("./AlignScore/src")
sys.path.append("./summac")
import torch
import nltk 
import textstat
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
from alignscore import AlignScore
from lens import download_model, LENS
from summac.model_summac import SummaCConv
import argparse
from f1chexbert import F1CheXbert
from radgraph import RadGraph, F1RadGraph
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import evaluate


nltk.download('punkt')

def calc_rouge(preds, refs):
  # Get ROUGE F1 scores
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                    use_stemmer=True, split_summaries=True)
  scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
  return (np.mean([s['rouge1'].fmeasure for s in scores])+ \
         np.mean([s['rouge2'].fmeasure for s in scores])+ \
         np.mean([s['rougeLsum'].fmeasure for s in scores]))/3.0

def cal_bleu(preds, refs):
  bleu = evaluate.load('sacrebleu')
  scores = bleu.compute(predictions=preds, references=refs)["score"]
  return scores

def cal_meteor(preds, refs):
  meteor = evaluate.load('meteor')
  scores = meteor.compute(predictions=preds, references=refs)["meteor"]
  return scores

def calc_bertscore(preds, refs):
  # Get BERTScore F1 scores
  P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cuda')#:0
  return np.mean(F1.tolist())

def calc_readability(preds):
  fkgl_scores = []
  cli_scores = []
  dcrs_scores = []
  for pred in preds:
    fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
    cli_scores.append(textstat.coleman_liau_index(pred))
    dcrs_scores.append(textstat.dale_chall_readability_score(pred))
  return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)

def calc_lens(preds, refs, docs):
  lens_path = download_model("davidheineman/lens")
  metric = LENS(lens_path, rescale=True)
  abstracts = [d.split("\n")[0] for d in docs]
  refs = [[x] for x in refs]

  scores = metric.score(abstracts, preds, refs, batch_size=8)#, gpus=1
  return np.mean(scores)

def calc_alignscore(preds, docs):
  model_path = hf_hub_download(repo_id="yzha/AlignScore", filename="AlignScore-base.ckpt")
  alignscorer = AlignScore(model='roberta-base', batch_size=16, device='cuda', \
                           ckpt_path=model_path, evaluation_mode='nli_sp')
  return np.mean(alignscorer.score(contexts=docs, claims=preds))

def cal_summac(preds, docs):
  model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
  return np.mean(model_conv.score(docs, preds)['scores'])

def cal_f1bert(preds, refs):
  f1chexbert = F1CheXbert()
  f1chexbert_score = f1chexbert(hyps=preds, refs=refs) 
  f1chexbert_score, _, _, _ = f1chexbert(hyps=preds, refs=refs) 
  return f1chexbert_score

def cal_radgraph(preds, refs):
  f1radgraph = F1RadGraph(reward_level="all")
  f1radgraph_score,  _, _, _ = f1radgraph(hyps=preds, refs=refs)
  return f1radgraph_score[-1]

def read_file_lines(path):
  with open(path, 'r') as f:
    lines = json.load(f)[:10]
  return lines

def cal_similarity(preds, refs):
  model = SentenceTransformer("all-MiniLM-L6-v2")
  scores = [np.array(model.similarity(model.encode(p), model.encode(r))) for p,r in zip(preds,refs)]
  return np.mean(scores)


def evaluate_all(pred_path, gold_path,task_name):
  # Load data from files
  refs_dicts = read_file_lines(gold_path)
  preds = read_file_lines(pred_path)
  # print(preds[0].keys)

  assert len(refs_dicts)==len(preds)

  refs = [d['reference'] for d in refs_dicts]  
  preds = [d['generated_caption'] for d in preds]
  if task_name == "lay_summ":
    docs = [d['document'] for d in refs_dicts]
  
  score_dict = {}

  # Relevance scores
  score_dict['ROUGE'] = calc_rouge(preds, refs)
  score_dict['BLEU'] = cal_bleu(preds, refs)
  score_dict['METEOR'] = cal_meteor(preds, refs)  
  score_dict['BERTScore'] = calc_bertscore(preds, refs)
  

  # # Readability scores
  fkgl_score, cli_score, dcrs_score = calc_readability(preds)
  score_dict['FKGL'] = fkgl_score
  score_dict['DCRS'] = dcrs_score
  score_dict['CLI'] = cli_score

  # Factuality scores
  if task_name == "lay_summ":
    score_dict['LENS'] = calc_lens(preds, refs, docs)
    score_dict['AlignScore'] = calc_alignscore(preds, docs)   
    score_dict['SummaC'] = cal_summac(preds, docs)
  else:
    score_dict['similarity'] = cal_similarity(preds, refs)
    score_dict["radgraph"] = cal_radgraph(preds,refs)
    score_dict["f1chexbert"] = cal_f1bert(preds,refs)

  print(score_dict)

  return score_dict

def write_scores(score_dict, output_filepath):
  # Write scores to file
  with open(output_filepath, 'w') as f:
    for key, value in score_dict.items():
      f.write(f"{key}: {value}\n")

def final_score(task_name):
  submit_dir = "./input_data"# ("elife.txt" and "plos.txt") with predictions
  truth_dir = "./reference_data"#directory with jsonl files containing references and articles 
  output_dir = "./output"

  if task_name == "lay_summ":
    # Calculate eLife scores
    elife_scores = evaluate_all(
      read_file_lines(os.path.join(submit_dir, 'elife.txt')), 
      read_file_lines(os.path.join(truth_dir, 'eLife_test.jsonl')),
      task_name
      )
    torch.cuda.empty_cache()

    # Calculate PLOS scores
    plos_scores = evaluate_all(
      read_file_lines(os.path.join(submit_dir, 'plos.txt')), 
      read_file_lines(os.path.join(truth_dir, 'PLOS_test.jsonl')),
      task_name
      )

    # Calculate overall scores
    final_scores = {key: np.mean([elife_scores[key], plos_scores[key]]) for key in elife_scores.keys()}
  elif task_name == "rrg":
    # Calculate RRG scores
    final_scores = evaluate_all(
      read_file_lines(os.path.join(submit_dir, 'open_rrg.txt')), 
      read_file_lines(os.path.join(truth_dir, 'OPEN_test.jsonl')),
      task_name
      )
  else:
    # Calculate RRG scores
    preds = read_file_lines(os.path.join(submit_dir, 'close_rrg.txt'))
    refs_dicts = read_file_lines(os.path.join(truth_dir, 'CLOSE_test.jsonl'))
    open_scores = evaluate_all(preds[:20000],refs_dicts[:20000],task_name)
    mimic_scores = evaluate_all(preds[20000:],refs_dicts[20000:],task_name)

    # Calculate overall scores
    final_scores = {key: np.mean([open_scores[key], mimic_scores[key]]) for key in open_scores.keys()}
    
  # Write overall score
  write_scores(final_scores, os.path.join(output_dir, 'scores.txt'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical text generation outputs.")
    parser.add_argument('--prediction_file', type=str, required=True,default= 'BioLaySumm2025-eLife_result.json', help='Path to the predictions JSON file.')
    parser.add_argument('--groundtruth_file', type=str, default= 'BioLaySumm2025-eLife_result.json',required=True, help='Path to the ground truth JSON file.')
    parser.add_argument('--task_name',  type=str,  default= 'lay_summ', required=True, help='The name of the task.') #"rrg","close_rrg"
    args = parser.parse_args()

    evaluate_all(args.prediction_file, args.groundtruth_file, args.task_name)
