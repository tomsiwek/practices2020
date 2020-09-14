import torch
import sol.pytorch as sol
from transformers import BertModel, BertConfig, BertTokenizer
import time

sol.set_seed(0)
device = torch.device("hip:0")

n_lines = 15000

#war_and_peace = []
with open("2600-0.txt","r") as book:
    war_and_peace = (line.rstrip() for line in book)
    war_and_peace = (line for line in war_and_peace if line)
    war_and_peace = list(war_and_peace)[:n_lines]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(war_and_peace, return_tensors="pt", padding=True, truncation=True)
model = BertModel.from_pretrained('bert-base-uncased')

input_ids = inputs["input_ids"]
vec_len = input_ids.size()[1]
#sol.cache.clear()
opt=sol.optimize(model, sol.input([n_lines, vec_len], dtype=torch.long))#, sol.input([0, 8], dtype=torch.long), sol.input([0, 8], dtype=torch.long), batch_size=1)
opt.load_state_dict(model.state_dict(), strict=False)
opt.to(device)

#help(model.forward)

#attention_mask = inputs["attention_mask"]
#token_type_ids = inputs["token_type_ids"]
print(input_ids.size())
#print(attention_mask.size())
#print(token_type_ids.size())
#print(inputs)
#inputs = inputs.to(device)
start_time = time.time()
opt.eval()
with torch.no_grad():
    input_ids = input_ids.to(device)
    #attention_mask = attention_mask.to(device)
    #token_type_ids = token_type_ids.to(device)
    #print(**inputs)
    outputs = opt(input_ids)#, attention_mask, token_type_ids)
    #outputs2 = model(input_ids=inputs["input_ids"])
    #outputs = outputs.cpu()
    #outputs=model(**inputs)
tot_time = time.time() - start_time
print("%.4f"%tot_time)
#print(outputs)
#print(outputs2)
