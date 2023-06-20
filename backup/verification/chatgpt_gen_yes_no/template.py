relation_list = ['country of citizenship', 'date of birth', 'place of birth', 'participant of',
                 'located in the administrative territorial entity', 'contains administrative territorial entity',
                 'participant', 'location', 'followed by', 'country', 'educated at', 'date of death', 'sibling',
                 'head of government', 'legislative body', 'conflict',
                 'applies to jurisdiction', 'instance of',
                 'performer', 'publication date', 'creator', 'author', 'composer', 'lyrics by', 'member of',
                 'notable work', 'inception', 'part of', 'cast member', 'director', 'has part', 'production company',
                 'owned by', 'headquarters location', 'developer', 'manufacturer', 'country of origin', 'publisher',
                 'parent organization', 'subsidiary', 'capital of', 'capital', 'spouse', 'father', 'child', 'religion',
                 'mother', 'located in or next to body of water', 'located on terrain feature', 'basin country',
                 'member of political party', 'mouth of the watercourse', 'place of death', 'military branch',
                 'work location', 'start time', 'award received', 'point in time', 'founded by', 'employer',
                 'head of state', 'member of sports team', 'league', 'present in work', 'position held', 'chairperson',
                 'languages spoken, written or signed', 'location of formation', 'operator', 'producer', 'record label',
                 'follows', 'replaced by', 'replaces', 'end time', 'subclass of', 'residence', 'sister city',
                 'original network', 'ethnic group', 'separated from', 'screenwriter', 'continent', 'platform',
                 'product or material produced', 'genre', 'series', 'narrative location', 'parent taxon',
                 'original language of work', 'dissolved, abolished or demolished', 'territory claimed by',
                 'characters', 'influenced by', 'official language', 'unemployment rate']

vf_template = {
    # "country of citizenship":
    #     {
    #         "country of citizenship": "In the sentence '{s[sentence]}', is it correct that the country of citizenship of {s[head]} is {s[tail]}?",
    #         "subject of (country)": "In the sentence '{s[sentence]}', is it correct that {s[head]} is a subject of {s[tail]}?",
    #         "citizenship": "In the sentence '{s[sentence]}', is it correct that {s[head]} holds citizenship of {s[tail]}?",
    #         "citizen of": "In the sentence '{s[sentence]}', is it correct that {s[head]} is a citizen of {s[tail]}? ",
    #         "national of": "In the sentence '{s[sentence]}', is it correct that {s[head]} is a national of {s[tail]}?",
    #         "(legal) nationality": "In the sentence '{s[sentence]}', is it correct that {s[head]} has {s[tail]} as their legal nationality?"
    #     }
    'country of citizenship': {
        'country of citizenship': "In the sentence '{s[sentence]}', is it correct that the country of citizenship of {s[head]} is {s[tail]}?",
        'citizenship': "Based on the sentence '{s[sentence]}', does {s[head]} hold citizenship in {s[tail]}?",
        'nationality': "Considering the sentence '{s[sentence]}', is the nationality of {s[head]} {s[tail]}?",
        'citizen of': "Using the sentence '{s[sentence]}', is {s[head]} a citizen of {s[tail]}?",
        'holds citizenship in': "With the sentence '{s[sentence]}', does {s[head]} hold citizenship in {s[tail]}?",
        'belongs to': "In the context of the sentence '{s[sentence]}', does {s[head]} belong to {s[tail]} as a citizen?",
        'passport country': "From the sentence '{s[sentence]}', is {s[tail]} the country that issued {s[head]}'s passport?",
        'registered as a citizen in': "Given the sentence '{s[sentence]}', is {s[head]} registered as a citizen in {s[tail]}?",
        'country of nationality': "In the sentence '{s[sentence]}', is {s[tail]} the country of nationality for {s[head]}?"
    },
}
