from src.keywords import KeywordMatcher

def words(text,gap=.2):return [{'word':s,'start':i*gap,'end':i*gap+.15,'probability':.95} for i,s in enumerate(text.split())]
def test_phrase_and_timestamps():
    h=KeywordMatcher().match(words('Please help me now'));assert len(h)==1 and h[0]['phrase']=='help me' and h[0]['start']==.2 and abs(h[0]['end']-.55)<1e-6
def test_short_words_exact_only():assert KeywordMatcher().match(words('hell shelf helpful'))==[]
def test_context_suppression():assert KeywordMatcher().match(words('help desk at the fire station'))==[]
def test_low_asr_confidence():
    w=words('help me');w[0]['probability']=.1;assert KeywordMatcher().match(w)==[]
def test_phrase_gap():assert not KeywordMatcher().match(words('save me',3))
def test_fuzzy_phrase():assert KeywordMatcher().match(words('leave me alon'))[0]['match']=='fuzzy'
