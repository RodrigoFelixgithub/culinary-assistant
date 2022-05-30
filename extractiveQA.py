from transformers import pipeline

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

class ExtractiveQA():

    def extractAnswer(self, question, context):
        QA_input = {
            'question': question,
            'context': context
        }
        return nlp(QA_input)
