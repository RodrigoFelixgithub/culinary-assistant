from urllib import response
from transitions import Machine
import json
from IPython.display import Image, HTML, display
import spacy as spacy
from spacy import displacy



class StateMachine():

    states = [
        { 'name': 'dummy'},
        { 'name': 'greeting', 'on_enter': ['greetingFunc'], 'on_exit': ['exitGreetingFunc']},
        { 'name': 'ask_for_desired_ingredients', 'on_enter': ['ask_for_desired_ingredientsFunc'], 'on_exit': ['exit_ask_for_desired_ingredientsFunc']},
        { 'name': 'ask_for_unwanted_ingredients', 'on_enter': ['ask_for_unwanted_ingredientsFunc'], 'on_exit': ['exit_ask_for_unwanted_ingredientsFunc']},
        { 'name': 'ask_for_keywords', 'on_enter': ['ask_for_keywordsFunc'], 'on_exit': ['exit_ask_for_keywordsFunc']},
        { 'name': 'ask_for_time_restrictions', 'on_enter': ['ask_for_time_restrictionsFunc'], 'on_exit': ['exit_ask_for_time_restrictionsFunc']},
        { 'name': 'show_top_recipes', 'on_enter': ['show_top_recipesFunc'], 'on_exit': ['exit_show_top_recipesFunc']},
        { 'name': 'skipIngredientsState', 'on_enter': ['ask_skip_ingredients']},
        { 'name': 'show_ingredients', 'on_enter': ['show_ingredientsFunc']},
        { 'name': 'show_steps', 'on_enter': ['show_stepsFunc']},
        { 'name': 'end', 'on_enter': ['endFunc']}
    ]
    


    def __init__(self, tokenizer, model, all_intents, searchEngine, qaExtractor):
        self.machine = Machine(model=self, states=self.states, initial='dummy')
        self.model = model
        self.tokenizer = tokenizer
        self.all_intents = all_intents
        self.searchEngine = searchEngine
        self.qaExtractor = qaExtractor

        self.userResponse = ''
        self.recipe = ''
        self.desired_ingredients = []
        self.unwanted_ingredients = []
        self.chosen_recipe = -1
        self.currentStep = 0
        self.keywords = []
        self.time_restriction = -1
        self.recipesarray = []
        self.keywordsPositive = []
        self.keywordsNegative = []

        self.nlp = spacy.load("en_core_web_sm")

        with open('../jsonData/recipesMapWithImages.json', "r") as read_file:
            self.recipesMap = json.load(read_file)

        # self.machine.add_transition(trigger='nomeDaTransicao_funcaoTransicao', source='estadoOrigem',
        # dest='estadoPosTransicao', before='funcaoAExecutarAntesDaTransicao', after='funcaoAExecutarDepoisDaTransicao',
        # conditions=['arrayDeCondicoesQueTeemQueSeConfimarQuandoSeChamaATransicao'])

        #dummy
        self.machine.add_transition(trigger='startStateMachine', source='dummy', dest='greeting')
        self.startStateMachine()

        #greeting
        self.machine.add_transition(trigger='IdentifyProcessIntent', source='greeting', dest='ask_for_desired_ingredients', before = 'defineRecipe')
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='greeting', dest='ask_for_desired_ingredients', before='defineRecipe')

        #ask_for_desired_ingredients
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='ask_for_desired_ingredients', dest='ask_for_unwanted_ingredients', before='define_desired_ingredients')
        self.machine.add_transition(trigger='IdentifyProcessIntent', source='ask_for_desired_ingredients', dest='ask_for_unwanted_ingredients', before='define_desired_ingredients')
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='ask_for_desired_ingredients', dest='ask_for_unwanted_ingredients')
        self.machine.add_transition(trigger='IngredientsConfirmationIntent', source='ask_for_desired_ingredients', dest='ask_for_unwanted_ingredients', before='define_desired_ingredients')


        #ask_for_unwanted_ingredients
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords')
        self.machine.add_transition(trigger='IdentifyProcessIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='QuestionIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='NextStepIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='IngredientsConfirmationIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')


        #ask_for_keywords
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='IngredientsConfirmationIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='StartStepsIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='AMAZON.SelectIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='ask_for_keywords', dest='ask_for_time_restrictions')

        #ask_for_time_restrictions
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='ask_for_time_restrictions', dest='show_top_recipes')
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='ask_for_time_restrictions', dest='show_top_recipes', before='define_time_restrictions')
        self.machine.add_transition(trigger='IngredientsConfirmationIntent', source='ask_for_time_restrictions', dest='show_top_recipes', before='define_time_restrictions')


        #show_top_recipes
        self.machine.add_transition(trigger='QuestionIntent', source='show_top_recipes', dest='skipIngredientsState', before='define_chosen_recipe')
        self.machine.add_transition(trigger='AMAZON.SelectIntent', source='show_top_recipes', dest='skipIngredientsState', before='define_chosen_recipe')
        self.machine.add_transition(trigger='IdentifyProcessIntent', source='show_top_recipes', dest='skipIngredientsState', before='define_chosen_recipe')


        #show_ingredients / skip ings
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='skipIngredientsState', dest='show_ingredients')
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='show_ingredients', dest='show_steps')
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='skipIngredientsState', dest='show_steps')
        
        #show_steps
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='show_steps', dest='end', conditions=['last_step'])
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='show_steps', dest='', before='define_next_step')

    

    def getIntent(self, text):
        input_encoding = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, max_length = 512, truncation = True)
        outputs = self.model(**input_encoding)

        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.all_intents[idx]

    def setUserResponse(self,userResponse):
        self.userResponse = userResponse



#greeting functions
    def greetingFunc(self):
        print('Hello! I\'m a cooking assistant and I\'m here to help you cook anything you want. What would you like to prepare today?')
    
    def defineRecipe(self): 
        myjson = self.qaExtractor.extractAnswer('Hello! I\'m a cooking assistant and I\'m here to help you cook anything you want. What would you like to prepare today?', self.userResponse)
        self.recipe = myjson['answer']
        print(self.recipe)

    def exitGreetingFunc(self): 
        print('Ok! I\'ll now ask you some questions to find the '+ self.recipe +' recipe you\'re looking for!')


#desired ings functions
    def ask_for_desired_ingredientsFunc(self):
        print('Are there any desired ingredients you\'d like the recipe to have? If so, could you enumerate them?')

    def define_desired_ingredients(self):
        myjson = self.qaExtractor.extractAnswer('Are there any desired ingredients you\'d like the recipe to have? If so, could you enumerate them?', self.userResponse)
        doc = self.nlp(myjson['answer'])
        for token in doc:
            if (not token.is_stop) and token.is_alpha and (token.pos_ == 'NOUN'):
                self.desired_ingredients.append(token.text)
        #self.desired_ingredients = ' '.join([token.text for token in doc if (not token.is_stop) and token.is_alpha and (token.pos_ == 'NOUN')])
        print(self.desired_ingredients)

    def get_desired_ingredients(self):
        return self.desired_ingredients

    def exit_ask_for_desired_ingredientsFunc(self):
        if(self.desired_ingredients == ''):
            print('Ok, I won\'t have to take into account any ingredient preference!')
        else:
            print('Ok, I\'ll take '+ ', '.join(self.desired_ingredients) + ' in consideration.') #todo mudar isto para fazer tomates, batatas e couves em vez de tomates, batatas, couves

    
#unwanted ings functions
    def ask_for_unwanted_ingredientsFunc(self): 
        print('Are there any ingredients you really don\'t want in the recipe? If so, could you also enumerate them?')

    def define_unwanted_ingredients(self):   
        myjson = self.qaExtractor.extractAnswer('Are there any ingredients you really don\'t want in the recipe? If so, could you also enumerate them?', self.userResponse)
        doc = self.nlp(myjson['answer'])
        for token in doc:
            if (not token.is_stop) and token.is_alpha and (token.pos_ == 'NOUN'):
                self.unwanted_ingredients.append(token.text)
        print(self.unwanted_ingredients)

    def get_unwanted_ingredients(self):    
        return self.unwanted_ingredients

    def exit_ask_for_unwanted_ingredientsFunc(self): 
        if(self.unwanted_ingredients == ''):
            print('I see, then I\'ll have no need to avoid any unwanted ingredients!')
        else:
            print('Got it, I\'ll find recipes which don\'t contain any '+ ', '.join(self.unwanted_ingredients) + '.') #todo mudar isto para fazer tomates, batatas e couves em vez de tomates, batatas, couves
        


#keywords functions
    def ask_for_keywordsFunc(self): 
        print('Should the recipe follow any dietary requirements?\nFor example, does it need to be vegan or gluten-free?')
    
    def define_keywords(self):
        myjson = self.qaExtractor.extractAnswer('Should the recipe follow any dietary requirements?\nFor example, does it need to be vegan or gluten-free?', self.userResponse)
        print(myjson['answer'])
        array = myjson['answer'].split()
        doc = self.nlp(myjson['answer'])
        for token in doc:
            if(token.is_stop):
                array.remove(token.text)
        

        for keyword in array:
            keywordDoc = self.nlp(keyword)
            keywordString = ' '.join([token.lemma_ for token in keywordDoc if not token.is_stop and token.is_alpha])
            if self.checkNegative(keyword):
                keywordCleaned = self.cleanNegativeWord(keywordString)
                self.keywordsNegative.append(keywordCleaned)
            else:
                self.keywordsPositive.append(keywordString)


        self.keywords = array
        print('keywords: ' + str(self.keywords))
        print('keywordsNegative: ' + str(self.keywordsNegative))
        print('keywordsPositive: ' + str(self.keywordsPositive))

    def get_keywords(self):
        return self.keywords
        
    def exit_ask_for_keywordsFunc(self): 
        if(len(self.keywords)==0):
            print('Understood! I won\'t factor in any dietary requirements.')
        else:
            print('Ok, I\'ll prioritize recipes that have '+ ', '.join(self.keywords) + ' requirements.') #todo mudar isto para fazer tomates, batatas e couves em vez de tomates, batatas, couves        

        
#time restriction funcs
    def ask_for_time_restrictionsFunc(self): 
        print('Last question! Are there any time restrictions for the food preparation? If so, how many minutes would you like it to take?')

    def define_time_restrictions(self):
        array = self.userResponse.split()
        self.time_restriction = [int(i) for i in array if i.isdigit()]
        if (len(self.time_restriction)>0): self.time_restriction = self.time_restriction[0]
        else: self.time_restriction = text2int(array[array.index('minutes') - 1])
        print(self.time_restriction)
  

    def get_time_restrictions(self):
        return self.time_restriction

    def exit_ask_for_time_restrictionsFunc(self): 
        if(self.time_restriction == -1):
            print('Very well, I won\'t take any time restriction into consideration.')
        else:
            print('Noted, I\'ll try to look for recipes that take less then '+ str(self.time_restriction) + ' minutes.') #todo mudar isto para fazer tomates, batatas e couves em vez de tomates, batatas, couves        

#top5 recipes funcs
    def show_top_recipesFunc(self):
        #myjson = self.searchEngine.queryOpenSearch('Holiday Salad', 10,None, None, ["salads"], ["lupine"], None)
        myjson=self.searchEngine.queryOpenSearch(self.recipe, 5, self.desired_ingredients, self.unwanted_ingredients, self.keywordsPositive, self.keywordsNegative, self.time_restriction)
        self.recipesarray = [recipe['fields']['recipeId'][0] for recipe in myjson['hits']['hits']]
        print('Done! I have found some recipes that fit your description.')
        for i in self.recipesarray:         #show recipes
            title = self.recipesMap[i]['recipe']['displayName']
            img = self.recipesMap[i]['recipe']['images'][0]['url']
            totalTime = str(self.recipesMap[i]['recipe']['totalTimeMinutes']) + ' minutes' if self.recipesMap[i]['recipe']['totalTimeMinutes'] != None else '-'
            rating = str(self.recipesMap[i]['rating']['ratingValue']) + '/5' if self.recipesMap[i]['rating'] != None else '-'
            displayResults(title,img,totalTime,rating)    

        print('Which one would you like to see?')
    
    def define_chosen_recipe(self):
        myjson = self.qaExtractor.extractAnswer('Which one would you like to see?', self.userResponse)

        self.chosen_recipe = self.extractChosenRecipe(self.recipesarray, myjson['answer'])

        #[int(i) for i in self.userResponse.split() if i.isdigit()][0] 
        #self.chosen_recipe = self.recipesarray[recipenumber-1]

    def get_chosen_recipe(self):
        return self.chosen_recipe

    def exit_show_top_recipesFunc(self): 
        print('You picked ' + self.recipesMap[self.chosen_recipe]['recipe']['displayName'] + '. Great choice!')

    def extractChosenRecipe(self, recipes, answer):
        length = len(recipes)
        if(length > 0 and any(s in answer for s in ['first', '1', 'one', self.recipesMap[recipes[0]]['recipe']['displayName']])) :
            return recipes[0]
        elif(length > 1 and any(s in answer for s in ['second', '2', 'two', self.recipesMap[recipes[1]]['recipe']['displayName']])) :
            return recipes[1]
        elif(length > 2 and any(s in answer for s in ['third', '3', 'three', self.recipesMap[recipes[2]]['recipe']['displayName']])) :
            return recipes[2]
        elif(length > 3 and any(s in answer for s in ['fourth', '4', 'four', self.recipesMap[recipes[3]]['recipe']['displayName']])) :
            return recipes[3]
        elif(length > 4 and any(s in answer for s in ['fifth', '5', 'five', self.recipesMap[recipes[4]]['recipe']['displayName']])) :
            return recipes[4]
        else:
            return -1


#skipingredientsstate
    def ask_skip_ingredients(self):
        print('Would you like to know the needed ingredients?')

#show ingredients func
    def show_ingredientsFunc(self):
        print('For this recipe you\'ll need the following ingredients:')
        #show ingredients
        ingredients = self.recipesMap[self.chosen_recipe]['recipe']['ingredients']
        # para ver o displayText fazer ingredients['displayText'] 
        #para ver o actual nome do ingrediente fazer ingredients['ingredient']  
        print('Are you ready to start cooking?')
    

#show steps func
    def show_stepsFunc(self):
        step = self.recipesMap[self.chosen_recipe]['recipe']['instructions'][self.currentStep]
        if(self.currentStep == 0): print('Ok! Let\'s begin!')
        print('Step n : _')
        #display(HTML(f"""
            #<div class ="row" style="margin-left:100px">
                #<img src="{img}" class="img-responsive" width="80px">
                #Step {n} : {text} <br>
            #</div>
        #"""))
        
    def define_next_step(self):
        self.currentStep = self.currentStep + 1

    def endFunc(self): print('Now that you\'ve finished cooking you can finally enjoy your meal! Bon appétit!')

    def checkNegative(self, keyword):
        return "free" in keyword or "no" in keyword

    def cleanNegativeWord(self, keyword):
        keyword = keyword.replace("free","").strip()
        keyword = keyword.replace("no ","").strip()
        return keyword

    @property
    def last_step(self):
        return self.currentStep == len(self.recipesMap[self.chosen_recipe]['recipe']['instructions']-1)


def displayResults(title, img, totalTime, rating):
    display(HTML(f"""
        <div class="row" style="display: flex; align-items: center; border-style: double;">
            <div class="column">
                <img src={img} style="width:200px; margin-right:20px"/>
            </div>
            <div class="column">
                <div class="row"><b>{title}</b> </div>
                <div class="row">Preparation time: {totalTime}</div>
                <div class="row">Rating: {rating}</div>
            </div>
        </div>
    """))


def text2int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
          raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current






