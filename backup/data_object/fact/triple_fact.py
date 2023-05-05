from .base_fact import *
import datetime
from ..mention import BaseMention


class TripleFact(BaseFact):
    """
    Store text-based triple evidence.
    """
    head = StringField(required=True)
    relationLabel = StringField(required=True)
    tail = StringField(required=True)

    headSpan = ListField(IntField(), required=True)
    relation = ReferenceField('BaseRelation', required=True)
    tailSpan = ListField(IntField())

    evidence = GenericReferenceField(required=True)
    evidenceText = StringField()
    verification = DictField()
    
    is_from_abstract = StringField()

    headWikidataEntity = ReferenceField('WikidataEntity')
    headWikipediaEntity = ReferenceField('WikipediaEntity')
    
    upVote = IntField()
    downVote = IntField()
    isNewFact = IntField()

    timestamp = DateTimeField()
    
    inPageId = ReferenceField('WikipediaPage')
    
    meta = {
        "collection": "triple_fact_v0_1_20220919",
        "db_alias": "NePtune"
    }
