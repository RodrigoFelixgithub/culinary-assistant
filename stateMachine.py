from transitions import Machine

class StateMachine():

    states = [
        { 'name': 'greeting', 'on_enter': ['greetingFunc']},
        { 'name': 'ask_for_extra_ingredients', 'on_enter': ['ask_for_extra_ingredientsFunc']},
        { 'name': 'ask_for_restricted_ingredients', 'on_enter': ['ask_for_restricted_ingredientsFunc']},
        { 'name': 'end', 'on_enter': ['greetingFunc']}
    ]
    
    def greetingFunc(self): print('aye wat recipe u want')
    def ask_for_extra_ingredientsFunc(self): print('more ingredients?')
    def ask_for_restricted_ingredientsFunc(self): print('do u want to exclude somre ingredients?')
    def endFunc(self): print('goodbye')


    def __init__(self, tokenizer, model, all_intents):
        self.machine = Machine(model=self, states=self.states, initial='greeting')
        self.model = model
        self.tokenizer = tokenizer
        self.all_intents = all_intents

        # self.machine.add_transition(trigger='nomeDaTransicao_funcaoTransicao', source='estadoOrigem',
        # dest='estadoPosTransicao', before='funcaoAExecutarAntesDaTransicao', after='funcaoAExecutarDepoisDaTransicao',
        # conditions=['arrayDeCondicoesQueTeemQueSeConfimarQuandoSeChamaATransicao'])

        self.machine.add_transition(trigger='AMAZON.YesIntent', source='greeting', dest='ask_for_extra_ingredients')
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='ask_for_extra_ingredients', dest='ask_for_restricted_ingredients')
        self.machine.add_transition(trigger='PreviousStepIntent', source='ask_for_extra_ingredients', dest='greeting')
        self.machine.add_transition(trigger='PreviousStepIntent', source='ask_for_restricted_ingredients', dest='greeting')
        self.machine.add_transition(trigger='AMAZON.StopIntent', source='*', dest='end')

    def getIntent(self, text):
        input_encoding = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, max_length = 512, truncation = True)
        outputs = self.model(**input_encoding)

        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.all_intents[idx]




