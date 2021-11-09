from transformers import BertTokenizer, BertModel
import torch
#from datasets import load_dataset
#dataset = load_dataset("imdb")
class embedReview():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.model = BertModel.from_pretrained("bert-large-cased")
    def embed(self,text):
        tokens = self.tokenizer.encode(text, add_special_tokens=True,max_length=512)
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            last_hidden_state = self.model(input_ids)[0]
        #take the final hidden state of the spe ccial classfication tolen [CLS]
        embedding = last_hidden_state[:,0,:]
        return embedding