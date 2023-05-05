from mongoengine import *


class BaseRelation(Document):
    """
    train_auto_glm_data object for relations in KGs; only refers to wikidata properties now.
    """
    # values
    text = StringField(required=True)
    description = StringField()
    alias = ListField(StringField())
    examples = ListField(StringField())
    # source
    source = StringField(required=True)
    sourceId = StringField()

    # constraints
    HeadConstraint = MapField(ListField(StringField()))
    TailConstraint = MapField(ListField(StringField()))

    meta = {
        "collection": "relation",
        # "indexes": [
        #     "sourceId",
        #     "$text"
        # ],
        "db_alias": "NePtune"
    }
