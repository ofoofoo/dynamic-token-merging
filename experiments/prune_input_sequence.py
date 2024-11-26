import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import random
import time
from fvcore.nn import FlopCountAnalysis

def prune_tokens(input_ids, embeddings, prune_n=2):
    input_ids = input_ids.to(embeddings.device)
    merged_input_ids_list = []
    merged_embeddings_list = []
    # PRUNE TOKEN N-WISE
    for i in range(0, input_ids.shape[1] - prune_n + 1, prune_n):
        merged_embedding = embeddings[:, i] # embeddings = [batch, seq_len, embed_dim]        
        merged_token_id = input_ids[:, i]
        merged_input_ids_list.append(merged_token_id)
        merged_embeddings_list.append(merged_embedding)
    
    remaining_tokens = input_ids.shape[1] % prune_n
    if remaining_tokens > 0:
        start_idx = input_ids.shape[1] - remaining_tokens
        if remaining_tokens == 1:
            merged_input_ids_list.append(input_ids[:, -1])
            merged_embeddings_list.append(embeddings[:, -1])
        else:
            merged_embedding = embeddings[:, start_idx]
            merged_token_id = input_ids[:, start_idx]
            merged_input_ids_list.append(merged_token_id)
            merged_embeddings_list.append(merged_embedding)
    
    merged_input_ids = torch.stack(merged_input_ids_list, dim=1)
    merged_embeddings = torch.stack(merged_embeddings_list, dim=1)
    
    return merged_input_ids, merged_embeddings

def modify_bart_for_token_merging(model, input_ids, prune_n=2):
    input_embeddings = model.get_input_embeddings()(input_ids) # SMALL NOTE: THIS FUNCTION DOES BOTH INPUT EMBEDDINGS + POSITIONAL EMBEDDINGS...
    print("INPUT EMEBDGS SHAEP")
    print(input_embeddings.shape)
    merged_input_ids, merged_embeddings = prune_tokens(input_ids, input_embeddings, prune_n) 
    return merged_input_ids


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)    
ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
test_dataset = ds['test']    
merge_n_values = [2, 3, 4]
for prune_n in merge_n_values:
    print(f"\nToken Merging with prune_n = {prune_n}")        
    subset_size = 50
    random_indices = random.sample(range(len(test_dataset)), subset_size)
    test_dataset_subset = test_dataset.select(random_indices)
    
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    
    total_flops = 0
    total_time = 0
    outputs = []
    targets = []        
    original_sequence_lengths = []
    merged_sequence_lengths = []
    
    for data in tqdm(test_dataset_subset):
        article = data['article']
        reference_summary = data['highlights']
        
        inputs = tokenizer(article, return_tensors='pt', max_length=1024, truncation=True).to(device)
        
        original_sequence_lengths.append(inputs['input_ids'].shape[1])            
        merged_input_ids = modify_bart_for_token_merging(model, inputs['input_ids'], prune_n)            
        merged_sequence_lengths.append(merged_input_ids.shape[1])
        
        flops_analysis = FlopCountAnalysis(model, merged_input_ids)
        total_flops += flops_analysis.total()
        
        torch.cuda.synchronize()
        start_time = time.time()
        summary_ids = model.generate(merged_input_ids)
        end_time = time.time()
        torch.cuda.synchronize()
        
        generated_summary = tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        total_time += (end_time - start_time)
        outputs.append(generated_summary)
        targets.append(reference_summary)
    
    average_flops = total_flops / len(test_dataset_subset)
    average_time = total_time / len(test_dataset_subset)
    
    avg_original_length = sum(original_sequence_lengths) / len(original_sequence_lengths)
    avg_merged_length = sum(merged_sequence_lengths) / len(merged_sequence_lengths)
    length_reduction_percentage = ((avg_original_length - avg_merged_length) / avg_original_length) * 100
    
    rouge_results = rouge.compute(predictions=outputs, references=targets)
    bleu_results = bleu.compute(predictions=outputs, references=targets)
    
    print("Inference Results:")
    print(f"Average FLOPs: {average_flops:.2e}")
    print(f"Average forward pass time: {average_time:.4f} seconds")
    print(f"Average Original Sequence Length: {avg_original_length:.2f}")
    print(f"Average Merged Sequence Length: {avg_merged_length:.2f}")
    print(f"Sequence Length Reduction: {length_reduction_percentage:.2f}%")
    print("ROUGE Scores:", rouge_results)
    print("BLEU Score:", bleu_results)