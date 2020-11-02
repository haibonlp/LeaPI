



class LexiconConceptRecognizer:
    def __init__(self, arg):
        self.arg = arg

        self.crdata = self.arg.crdata


    def get_field_concepts(self, text):
        concepts = set()
        if len(text.strip()) <=0:
            return concepts
        if len(text.split())==1 and text.startswith('@'): # pronouns
            concepts.add('person')
            return concepts
        if text in  self.crdata.inst2concepts:
            concepts = self.crdata.inst2concepts[text]
            return concepts
        concepts.add('#unknow#')
        return concepts


    def get_event_field_concepts(self, event):
        fieldnamelist = ['Agent', 'Predicate', 'Theme', 'PP']
        event.field2concepts = {}
        for fname in fieldnamelist:
            text = event.get_value_by_field(fname)
            if fname == 'PP':
                text = event.remove_preposition_in_PP()
            #text = event.remove_possessive_pronoun(text)
            concepts = self.get_field_concepts(text)
            event.field2concepts[fname] = concepts

        #print(f"    Event : {event.field2value}")
        #print(f" Concepts : {event.field2concepts}\n")




class ConceptRecognizer:
    def __init__(self, arg):
        self.arg = arg
        self.recognizer = LexiconConceptRecognizer(arg)


    def recognize_event_concepts(self, eventlist):
        for event in eventlist:
            self.identify_event_field_concepts(event)


    def identify_event_field_concepts(self, event):

        self.recognizer.get_event_field_concepts(event)






