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
        self.chosen_recipe = ''
        self.currentStep = 0
        self.keywords = []
        self.time_restriction = -1
        self.recipesarray = []
        self.keywordsPositive = []
        self.keywordsNegative = []

        self.nlp = spacy.load("en_core_web_sm")

        with open('../jsonData/recipesMapWithImages.json', "r") as read_file:
            self.recipesMap = json.load(read_file)


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
        self.machine.add_transition(trigger='AMAZON.SelectIntent', source='ask_for_desired_ingredients', dest='ask_for_unwanted_ingredients', before='define_desired_ingredients')
        self.machine.add_transition(trigger='NextStepIntent', source='ask_for_desired_ingredients', dest='ask_for_unwanted_ingredients', before='define_desired_ingredients')
        self.machine.add_transition(trigger='StartStepsIntent', source='ask_for_desired_ingredients', dest='ask_for_unwanted_ingredients', before='define_desired_ingredients')
        self.machine.add_transition(trigger='QuestionIntent', source='ask_for_desired_ingredients', dest='ask_for_unwanted_ingredients', before='define_desired_ingredients')


        self.machine.add_transition(trigger='AMAZON.FallbackIntent', source='ask_for_desired_ingredients', dest='greeting', before='reset_desired_ingredients', conditions=['user_said_back'])
        self.machine.add_transition(trigger='PreviousStepIntent', source='ask_for_desired_ingredients', dest='greeting', before='reset_desired_ingredients', conditions=['user_said_back'])


        #ask_for_unwanted_ingredients
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords')
        self.machine.add_transition(trigger='IdentifyProcessIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='QuestionIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='NextStepIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='IngredientsConfirmationIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='AMAZON.SelectIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
        self.machine.add_transition(trigger='StartStepsIntent', source='ask_for_unwanted_ingredients', dest='ask_for_keywords', before='define_unwanted_ingredients')
       
        self.machine.add_transition(trigger='AMAZON.FallbackIntent', source='ask_for_unwanted_ingredients', dest='ask_for_desired_ingredients', before='reset_unwanted_ingredients',conditions=['user_said_back'])
        self.machine.add_transition(trigger='PreviousStepIntent', source='ask_for_unwanted_ingredients', dest='ask_for_desired_ingredients', before='reset_unwanted_ingredients',conditions=['user_said_back'])
        
        #ask_for_keywords
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='IngredientsConfirmationIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='StartStepsIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='AMAZON.SelectIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='ask_for_keywords', dest='ask_for_time_restrictions')
        self.machine.add_transition(trigger='IdentifyProcessIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='NextStepIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')
        self.machine.add_transition(trigger='QuestionIntent', source='ask_for_keywords', dest='ask_for_time_restrictions', before='define_keywords')

        self.machine.add_transition(trigger='PreviousStepIntent', source='ask_for_keywords', dest='ask_for_unwanted_ingredients', before='reset_keywords',conditions=['user_said_back'])
        self.machine.add_transition(trigger='AMAZON.FallbackIntent', source='ask_for_keywords', dest='ask_for_unwanted_ingredients', before='reset_keywords',conditions=['user_said_back'])

        #ask_for_time_restrictions
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='ask_for_time_restrictions', dest='show_top_recipes')
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='ask_for_time_restrictions', dest='show_top_recipes', before='define_time_restrictions')
        self.machine.add_transition(trigger='IngredientsConfirmationIntent', source='ask_for_time_restrictions', dest='show_top_recipes', before='define_time_restrictions')
        self.machine.add_transition(trigger='NextStepIntent', source='ask_for_time_restrictions', dest='show_top_recipes', before='define_time_restrictions')
        self.machine.add_transition(trigger='IdentifyProcessIntent', source='ask_for_time_restrictions', dest='show_top_recipes', before='define_time_restrictions')
        self.machine.add_transition(trigger='AMAZON.SelectIntent', source='ask_for_time_restrictions', dest='show_top_recipes', before='define_time_restrictions')

        self.machine.add_transition(trigger='PreviousStepIntent', source='ask_for_time_restrictions', dest='ask_for_keywords', before='reset_time_restrictions',conditions=['user_said_back'])
        self.machine.add_transition(trigger='AMAZON.FallbackIntent', source='ask_for_time_restrictions', dest='ask_for_keywords', before='reset_time_restrictions',conditions=['user_said_back'])

        #show_top_recipes
        self.machine.add_transition(trigger='QuestionIntent', source='show_top_recipes', dest='skipIngredientsState', before='define_chosen_recipe')
        self.machine.add_transition(trigger='AMAZON.SelectIntent', source='show_top_recipes', dest='skipIngredientsState', before='define_chosen_recipe')
        self.machine.add_transition(trigger='IdentifyProcessIntent', source='show_top_recipes', dest='skipIngredientsState', before='define_chosen_recipe')
        self.machine.add_transition(trigger='NextStepIntent', source='show_top_recipes', dest='skipIngredientsState', before='define_chosen_recipe')
        self.machine.add_transition(trigger='GoToStepIntent', source='show_top_recipes', dest='skipIngredientsState', before='define_chosen_recipe')

        self.machine.add_transition(trigger='PreviousStepIntent', source='show_top_recipes', dest='ask_for_time_restrictions', before='reset_chosen_recipe',conditions=['user_said_back'])
        self.machine.add_transition(trigger='AMAZON.FallbackIntent', source='show_top_recipes', dest='ask_for_time_restrictions', before='reset_chosen_recipe',conditions=['user_said_back'])

        #show_ingredients / skip ings
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='skipIngredientsState', dest='show_ingredients')
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='show_ingredients', dest='show_steps')
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='skipIngredientsState', dest='show_steps')
        
        self.machine.add_transition(trigger='PreviousStepIntent', source='skipIngredientsState', dest='show_top_recipes',conditions=['user_said_back'])
        self.machine.add_transition(trigger='AMAZON.FallbackIntent', source='skipIngredientsState', dest='show_top_recipes',conditions=['user_said_back'])

        #show_steps
        self.machine.add_transition(trigger='AMAZON.NoIntent', source='show_steps', dest='show_steps')
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='show_steps', dest='end', conditions=['last_step'])
        self.machine.add_transition(trigger='AMAZON.YesIntent', source='show_steps', dest='show_steps', before='define_next_step')
        self.machine.add_transition(trigger='NextStepIntent', source='show_steps', dest='end', conditions=['last_step'])
        self.machine.add_transition(trigger='NextStepIntent', source='show_steps', dest='show_steps', before='define_next_step')        
    
        self.machine.add_transition(trigger='PreviousStepIntent', source='show_steps', dest='show_top_recipes', conditions=['first_step'])        
        self.machine.add_transition(trigger='AMAZON.FallbackIntent', source='show_steps', dest='show_top_recipes', conditions=['first_step'])        
        self.machine.add_transition(trigger='PreviousStepIntent', source='show_steps', dest='show_steps', before='define_prev_step')        
        self.machine.add_transition(trigger='AMAZON.FallbackIntent', source='show_steps', dest='show_steps', before='define_prev_step')  


        #general transition
        self.machine.add_transition(trigger='AMAZON.StopIntent', source='*', dest='greeting', before='reset_all_vars', conditions=['user_said_stop'])  
        self.machine.add_transition(trigger='AMAZON.CancelIntent', source='*', dest='greeting', before='reset_all_vars', conditions=['user_said_stop'])  


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
        self.recipe = ''
        print('Hello! I\'m a cooking assistant and I\'m here to help you cook anything you want. What would you like to prepare today?')
    
    def defineRecipe(self): 
        myjson = self.qaExtractor.extractAnswer('Hello! I\'m a cooking assistant and I\'m here to help you cook anything you want. What would you like to prepare today?', self.userResponse)
        self.recipe = myjson['answer']
        #print(self.recipe)
    
    def resetRecipe(self): 
        self.recipe = ''

    def exitGreetingFunc(self): 
        if self.recipe != '': print('Ok! I\'ll now ask you some questions to find the '+ self.recipe +' recipe you\'re looking for!')


#desired ings functions
    def ask_for_desired_ingredientsFunc(self):
        self.desired_ingredients = []
        print('Are there any desired ingredients you\'d like the recipe to have? If so, could you enumerate them?')

    def define_desired_ingredients(self):
        myjson = self.qaExtractor.extractAnswer('Are there any desired ingredients you\'d like the recipe to have? If so, could you enumerate them?', self.userResponse)
        #print(myjson['score'])
        if myjson['score'] < 0.0001 :
            raise Exception("Illegal score: " + str(myjson['score']))
        doc = self.nlp(myjson['answer'])
        #print(myjson['answer'])
        for token in doc:
            #print(str(token.text) + '   ' + str(token.pos_))
            if (not token.is_stop) and token.is_alpha and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN'):
                self.desired_ingredients.append(token.text)
        #self.desired_ingredients = ' '.join([token.text for token in doc if (not token.is_stop) and token.is_alpha and (token.pos_ == 'NOUN')])
        #print(self.desired_ingredients)
        if (len(self.desired_ingredients) == 0):
            raise Exception("Illegal desired_ingredients size: " + str(len(self.desired_ingredients)))
    
    def reset_desired_ingredients(self): 
        self.desired_ingredients = -1

    def get_desired_ingredients(self):
        return self.desired_ingredients

    def exit_ask_for_desired_ingredientsFunc(self):
        if self.desired_ingredients != -1:
            if(len(self.desired_ingredients) == 0):
                print('Ok, I won\'t have to take into account any ingredient preference!')
            elif(len(self.desired_ingredients) == 1):
                print('Ok, I\'ll take ' + self.desired_ingredients[0] + ' in consideration.') 
            else:
                print('Ok, I\'ll take '+ ', '.join(self.desired_ingredients[:-1]) + ' and '+ self.desired_ingredients[-1] + ' in consideration.')

#unwanted ings functions
    def ask_for_unwanted_ingredientsFunc(self): 
        self.unwanted_ingredients = []
        print('Are there any ingredients you really don\'t want in the recipe? If so, could you also enumerate them?')

    def define_unwanted_ingredients(self):   
        myjson = self.qaExtractor.extractAnswer('Are there any ingredients you really don\'t want in the recipe? If so, could you also enumerate them?', self.userResponse)
        #(myjson['score'])
        if myjson['score'] < 0.0001 :
            raise Exception("Illegal score: " + str(myjson['score']))
        doc = self.nlp(myjson['answer'])
        for token in doc:
            if (not token.is_stop) and token.is_alpha and (token.pos_ == 'NOUN'):
                self.unwanted_ingredients.append(token.text)
        #print(self.unwanted_ingredients)
        if (len(self.unwanted_ingredients) == 0):
            raise Exception("Illegal unwanted_ingredients size: " + str(len(self.unwanted_ingredients)))

    def reset_unwanted_ingredients(self): 
        self.unwanted_ingredients = -1


    def get_unwanted_ingredients(self):    
        return self.unwanted_ingredients

    def exit_ask_for_unwanted_ingredientsFunc(self): 
        if self.unwanted_ingredients != -1:
            if(len(self.unwanted_ingredients) == 0):
                print('I see, then I\'ll have no need to avoid any unwanted ingredients!')
            elif(len(self.unwanted_ingredients) == 1):
                print('Got it, I\'ll find recipes which don\'t contain any '+ self.unwanted_ingredients[0] + '.') 
            else:
                print('Got it, I\'ll find recipes which don\'t contain any '+ ', '.join(self.unwanted_ingredients[:-1]) + ' and '+ self.unwanted_ingredients[-1] + '.')        

#keywords functions
    def ask_for_keywordsFunc(self): 
        self.keywords = []
        print('Should the recipe follow any dietary requirements?\nFor example, does it need to be vegan or gluten-free?')
    
    def define_keywords(self):
        myjson = self.qaExtractor.extractAnswer('Should the recipe follow any dietary requirements?\nFor example, does it need to be vegan or gluten-free?', self.userResponse)
        if myjson['score'] < 0.0001 :
            raise Exception("Illegal score: " + str(myjson['score']))
        sentence = myjson['answer'].strip(",.:;")
        #print(sentence)
        array = sentence.split()
        doc = self.nlp(myjson['answer'])
        for token in doc:
            if(token.is_stop):
                array.remove(token.text)

        if (len(array) == 0):
            raise Exception("Illegal keywords size: " + str(len(array)))        

        for keyword in array:
            keywordDoc = self.nlp(keyword)
            keywordString = ' '.join([token.lemma_ for token in keywordDoc if not token.is_stop and token.is_alpha])
            if self.checkNegative(keyword):
                keywordCleaned = self.cleanNegativeWord(keywordString)
                self.keywordsNegative.append(keywordCleaned)
            else:
                self.keywordsPositive.append(keywordString)

        self.keywords = array
        #print('keywords: ' + str(self.keywords))
        #print('keywordsNegative: ' + str(self.keywordsNegative))
        #print('keywordsPositive: ' + str(self.keywordsPositive))

    def get_keywords(self):
        return self.keywords

    def reset_keywords(self): 
        self.keywords = -1
        
    def exit_ask_for_keywordsFunc(self): 
        if self.keywords != -1:
            if(len(self.keywords) == 0):
                print('Understood! I won\'t factor in any dietary requirements.')
            elif(len(self.keywords) == 1):
                print('Ok, I\'ll prioritize recipes that have '+ self.keywords[0] + ' requirements.')
            else:
                print('Ok, I\'ll prioritize recipes that have '+ ', '.join(self.keywords[:-1]) + ' and '+ self.keywords[-1] + ' requirements.')  
            
#time restriction funcs
    def ask_for_time_restrictionsFunc(self): 
        self.time_restriction = -1
        print('Last question! Are there any time restrictions for the food preparation? If so, how many minutes would you like it to take?')

    def define_time_restrictions(self):
        array = self.userResponse.split()
        self.time_restriction = [int(i) for i in array if i.isdigit()]
        if (len(self.time_restriction)>0): self.time_restriction = self.time_restriction[0]
        else: self.time_restriction = text2int(array[array.index('minutes') - 1])
        #print(self.time_restriction)

    def reset_time_restrictions(self): 
        self.time_restriction = -2

    def get_time_restrictions(self):
        return self.time_restriction

    def exit_ask_for_time_restrictionsFunc(self): 
        if(self.time_restriction == -1):
            print('Very well, I won\'t take any time restriction into consideration.')
        elif self.time_restriction >= 0:
            print('Noted, I\'ll try to look for recipes that take less then '+ str(self.time_restriction) + ' minutes.') #todo mudar isto para fazer tomates, batatas e couves em vez de tomates, batatas, couves        

#top5 recipes funcs
    def show_top_recipesFunc(self):
        self.chosen_recipe = ''
        #myjson = self.searchEngine.queryOpenSearch('Holiday Salad', 10,None, None, ["salads"], ["lupine"], None)
        myjson=self.searchEngine.queryOpenSearch(self.recipe, 5, self.desired_ingredients, self.unwanted_ingredients, self.keywordsPositive, self.keywordsNegative, self.time_restriction)
        self.recipesarray = [recipe['fields']['recipeId'][0] for recipe in myjson['hits']['hits']]
        if len(self.recipesarray) == 0: print('sorry but none of my recipes match your description')
        else: 
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

    def get_chosen_recipe(self):
        return self.chosen_recipe
    
    def reset_chosen_recipe(self):
        self.chosen_recipe = -1

    def exit_show_top_recipesFunc(self): 
        if self.chosen_recipe != -1 : print('You picked ' + self.recipesMap[self.chosen_recipe]['recipe']['displayName'] + '. Great choice!')

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
        displayIngredients(ingredients)

        print('Are you ready to start cooking?')
    

#show steps func
    def show_stepsFunc(self):
        instructions = self.recipesMap[self.chosen_recipe]['recipe']['instructions']
        step = instructions[self.currentStep]
        img = step['stepImages'][0]['url']
        if step['stepText'] is not None and len(step['stepText']) > 0 :
            text = step['stepText']
        else: text = ''
        if step['stepTitle'] is not None and len(step['stepTitle']) > 0 :
            title = step['stepTitle']
        else:
            title = ''
        if(self.currentStep == 0):
            print('Ok! Let\'s begin!\n')
        displayStep(img, self.currentStep+1, title, text)
        if(self.currentStep < len(instructions)-1): print('Would you like to proceed to the next step?')
        else: print('This was the last step, would you like to proceed?')
        
    def define_next_step(self):
        self.currentStep = self.currentStep + 1

    def define_prev_step(self):
        self.currentStep = self.currentStep - 1       

    def endFunc(self): print('Now that you\'ve finished cooking you can finally enjoy your meal! Bon appétit!')

    def reset_all_vars(self):
        self.userResponse = ''
        self.recipe = -1
        self.desired_ingredients = -1
        self.unwanted_ingredients = -1
        self.chosen_recipe = -1
        self.currentStep = 0
        self.keywords = -1
        self.time_restriction = -2
        self.recipesarray = []
        self.keywordsPositive = []
        self.keywordsNegative = []

    def checkNegative(self, keyword):
        return "free" in keyword or "no" in keyword

    def cleanNegativeWord(self, keyword):
        keyword = keyword.replace("free","").strip()
        keyword = keyword.replace("no ","").strip()
        return keyword

    @property
    def last_step(self):
        return self.currentStep == len(self.recipesMap[self.chosen_recipe]['recipe']['instructions']) -1

    @property
    def first_step(self):
        return self.currentStep == 0

    @property
    def user_said_back(self):
        if not any(s in self.userResponse.lower() for s in ['back', 'previous']):
        #if not ('back' in  or 'Back' in self.userResponse or 'previous'  in self.userResponseor or 'Previous'  in self.userResponse):
            raise Exception('user didnt say back or previous')
        return True
    
    @property
    def user_said_stop(self):
        if not any(s in self.userResponse.lower() for s in ['stop', 'cancel']):
            raise Exception('user didnt say back or previous')
        return True



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

def displayResults(title, img, totalTime, rating):
    display(HTML(f"""
        <div style="display: flex; align-items: center; border-style: double;">
            <div>
                <img src={img} style="width:200px; margin-right:20px"/>
            </div>
            <div>
                <div><b>{title}</b> </div>
                <div>Preparation time: {totalTime}</div>
                <div>Rating: {rating}</div>
            </div>
        </div>
    """))

def displayIngredients(ingredients):
    ingDisplay = ""

    for ing in ingredients:
        ingDisplay = ingDisplay +'<li style="padding-bottom:3px;"> '+ ing['displayText']
        
    display(HTML(f"""
        <div style="align-items: center;border-style: double;">
            <h2 style="text-align:center;">Ingredient List</h3>
            <ul style="font-size: 17px;">{ingDisplay}</ul>
        </div>
    """))

def displayStep(img, number, title, description):
    display(HTML(f"""
        <div style="display: flex; align-items: center; border-style: double;">
            <div>
                <img src={img} style="width:200px; margin-right:20px"/>
            </div>
            <div>
                <div><b>Step {number}: {title}</b> </div>
                <div>{description}</div>
            </div>
        </div>
    """))






