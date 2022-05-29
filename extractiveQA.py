from transformers import pipeline

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

class ExtractiveQA():

    def extractAnswer(self, question, response):
        QA_input = {
            'question': question,
            'context': response
        }
        return nlp(QA_input)
