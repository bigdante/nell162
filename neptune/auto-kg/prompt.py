relation_list = ['country of citizenship', 'date of birth', 'place of birth', 'participant of', 'located in the administrative territorial entity',
                 'contains administrative territorial entity', 'participant', 'location', 'followed by', 'country', 'educated at', 'date of death', 'sibling',
                 'head of government', 'legislative body', 'conflict', 'applies to jurisdiction', 'instance of', 'performer', 'publication date', 'creator', 'author', 'composer',
                 'lyrics by', 'member of', 'notable work', 'inception', 'part of', 'cast member', 'director', 'has part', 'production company', 'owned by', 'headquarters location',
                 'developer', 'manufacturer', 'country of origin', 'publisher', 'parent organization', 'subsidiary', 'capital of', 'capital', 'spouse', 'father', 'child',
                 'religion', 'mother', 'located in or next to body of water', 'located on terrain feature', 'basin country', 'member of political party',
                 'mouth of the watercourse', 'place of death', 'military branch', 'work location', 'start time', 'award received', 'point in time', 'founded by', 'employer',
                 'head of state', 'member of sports team', 'league', 'present in work', 'position held', 'chairperson', 'languages spoken, written or signed',
                 'location of formation', 'operator', 'producer', 'record label', 'follows', 'replaced by', 'replaces', 'end time', 'subclass of', 'residence', 'sister city',
                 'original network', 'ethnic group', 'separated from', 'screenwriter', 'continent', 'platform', 'product or material produced', 'genre', 'series',
                 'narrative location', 'parent taxon', 'original language of work', 'dissolved, abolished or demolished', 'territory claimed by', 'characters', 'influenced by',
                 'official language'
                 ]

extract_prompt = "You are a fact extractor. You should answer the question with the given sentences.\n" \
                 "\n" \
                 "Constraints:\n" \
                 "1.Your answers can only be words from the given list in the sentence.\n" \
                 "2.You should output all the possible answers.\n" \
                 "\n" \
                 "Response Format: \n" \
                 "You must only respond in list format as described: ['answer1','answer2',....]\n" \
                 "(If no correct option is found, output a list: ['unknown'])\n" \
                 "Ensure the response can be parsed by Python eval()"

re_alias_prompt = "Alias templates below do not perform well: \n" \
                  "{s[template]}\n" \
                  "Different alias templates performance scores are:{s[score]}\n" \
                  "Note: tp is true positive, fp is false positive.\n" \
                  "The {s[relation]} descript and examples are : \n" \
                  "{s[relation_descript]}\n" \
                  "Save the well-performed alias and abandon the bad ones, you should always reserve the first one.\n" \
                  "Give me at least 8 more generalized alias template according to the descript.\n" \
                  "Constraints: \n" \
                  "1.You should not use the descript words directly.\n" \
                  "2.Do not use the bad used alias.\n" \
                  "3.Template should contain the alias words" \
                  "Response Format: \n" \
                  "Your respond must be a dict format, for example: \n" \
                  "    {{\n" \
                  "      'alias1':'template1'\n" \
                  "      'alias2':'template2'\n" \
                  "      'alias3':'template3'\n" \
                  "    }}" \
                  "\n" \
                  "Ensure the response can be parsed by Python eval().\n"
