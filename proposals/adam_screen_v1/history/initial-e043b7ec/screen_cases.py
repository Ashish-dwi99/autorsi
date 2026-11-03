"""Frozen new browser-rendered desktop-like/mobile-like fixtures, not OS tasks.

The Python expected result is withheld from policies. Fixture JS records ordinary
UI receipts without comparing them to the requested goal. No model success flag.
Verification uses distinct fixed seeds and the opposite layout per family.
"""
from dataclasses import dataclass
import hashlib
import html
import json
import random

FAMILIES = ("contact", "checklist", "choice", "dialog", "tabs", "counter", "scroll", "search")
SEEDS = {"development": 631047, "verification": 975263}
VIEWPORTS = {"desktop-like": (1024, 768), "mobile-like": (390, 844)}


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    family: str
    layout: str
    seed: int
    goal: str
    html: str
    expected_json: str

    @property
    def viewport(self):
        w, h = VIEWPORTS[self.layout]
        return {"width": w, "height": h}

    @property
    def fixture_sha256(self):
        return hashlib.sha256(self.html.encode()).hexdigest()

    @property
    def expected(self):
        return json.loads(self.expected_json)

    def manifest(self):
        return {"case_id": self.case_id, "split": self.split, "family": self.family,
                "layout": self.layout, "seed": self.seed, "viewport": self.viewport,
                "fixture_sha256": self.fixture_sha256,
                "goal_sha256": hashlib.sha256(self.goal.encode()).hexdigest(),
                "oracle_sha256": hashlib.sha256(self.expected_json.encode()).hexdigest()}


STYLE = """*{box-sizing:border-box}html,body{margin:0;font:18px Arial,sans-serif;background:#eef2f8;color:#172235}
header{background:#172235;color:white;padding:18px 24px}header small{display:block;color:#c4d4ee;font-size:13px;margin-top:5px}
main{max-width:850px;margin:22px auto;padding:0 20px 30px}h1{font-size:27px;margin:0 0 18px}h2{font-size:21px}
.panel,.card{background:white;padding:22px;border:1px solid #bdcada;border-radius:12px;margin:14px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid .card{margin:0}label{display:block;margin:14px 0}
input[type=text],input[type=search]{font:inherit;border:2px solid #8293ad;padding:12px;width:100%;border-radius:7px}
input[type=checkbox],input[type=radio]{width:23px;height:23px;vertical-align:middle;margin-right:12px}
button{font:inherit;background:#275dc7;color:white;padding:12px 18px;border:2px solid transparent;border-radius:8px;cursor:pointer;margin:5px}
button.secondary{background:#e5eaf3;color:#172235}button:focus,input:focus{outline:3px solid #df8e16;outline-offset:2px}
output{display:block;background:#e7f4e8;color:#185423;padding:12px;margin-top:18px;white-space:pre-wrap;min-height:24px}
output:empty{display:none}.row{display:flex;align-items:center;justify-content:space-between;gap:8px}.row label{flex:1}
dialog{max-width:340px;border:2px solid #526b92;border-radius:14px;padding:24px}dialog::backdrop{background:#0008}
[hidden]{display:none!important}.spacer{height:420px}.count{font-size:40px;text-align:center;padding:15px}
@media(max-width:600px){main{margin:18px auto;padding:0 14px 25px}.grid{grid-template-columns:1fr}.panel,.card{padding:18px}header{padding:14px}h1{font-size:25px}button{padding:11px 13px}}
"""


def _page(title, content, script, accent):
    # The only receipt writer is normal fixture event handling. No expected
    # answer, reference action coordinates or success oracle is embedded here.
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<style>{STYLE}button{{background:{accent}}}</style></head><body>
<header>Screen Studio<small>Isolated browser fixture</small></header><main><h1>{html.escape(title)}</h1>
{content}<output id="receipt" aria-live="polite"></output></main><script>
window.__screenReceipt=null;
function record(value){{window.__screenReceipt=JSON.parse(JSON.stringify(value));document.getElementById('receipt').textContent='Saved: '+JSON.stringify(value);}}
{script}
</script></body></html>"""


def _build(split, index):
    family = FAMILIES[index]
    seed = SEEDS[split] + index * 101
    rng = random.Random(seed)
    layout = ("desktop-like" if (index + (split == "verification")) % 2 == 0 else "mobile-like")
    names = ["Mira", "Theo", "Nora", "Evan", "Aria", "Luca", "Iris", "Omar"]
    words = (["Maple", "Cedar", "Coral", "Amber", "Indigo", "Olive", "Silver", "Ruby"]
             if split == "development" else ["Lilac", "Pearl", "Slate", "Jade", "Hazel", "Azure", "Onyx", "Saffron"])
    rng.shuffle(words)
    target = words[1]
    esc = html.escape
    js = lambda x: json.dumps(x)
    if family == "contact":
        person = rng.choice(names) + " " + str(rng.randrange(20, 90))
        code = "R" + str(rng.randrange(100, 999))
        goal = f"Replace the contact name with {person}, enter ticket {code}, and save the contact."
        content = '<form id="form" class="panel"><label>Contact name<input id="name" type="text" value="Draft"></label><label>Ticket<input id="ticket" type="text"></label><button>Save contact</button><button type="button" class="secondary" id="cancel">Cancel</button></form>'
        script = "document.getElementById('form').onsubmit=e=>{e.preventDefault();record({name:document.getElementById('name').value,ticket:document.getElementById('ticket').value});};document.getElementById('cancel').onclick=()=>record({cancelled:true});"
        expected = {"name": person, "ticket": code}
    elif family == "checklist":
        selected = sorted((words[0], words[2]))
        goal = f"Select only {selected[0]} and {selected[1]} notifications, then save preferences."
        content = '<form id="form" class="panel"><h2>Notifications</h2>' + ''.join(f'<label><input type="checkbox" name="notice" value="{esc(w)}">{esc(w)}</label>' for w in words[:4]) + '<button>Save preferences</button></form>'
        script = "document.getElementById('form').onsubmit=e=>{e.preventDefault();record({selected:[...document.querySelectorAll('input:checked')].map(x=>x.value).sort()});};"
        expected = {"selected": selected}
    elif family == "choice":
        goal = f"Choose the {target} delivery plan and confirm the selection."
        cards = ''.join(f'<label class="card"><input type="radio" name="plan" value="{esc(w)}">{esc(w)}<p>Delivery plan {n+1}</p></label>' for n,w in enumerate(words[:4]))
        content = '<form id="form"><div class="grid">'+cards+'</div><button>Confirm selection</button></form>'
        script = "document.getElementById('form').onsubmit=e=>{e.preventDefault();record({plan:document.querySelector('input:checked')?.value||null});};"
        expected = {"plan": target}
    elif family == "dialog":
        goal = f"Archive the {target} project and confirm in the dialog."
        content = '<div class="panel">' + ''.join(f'<div class="row"><span>{esc(w)}</span><button data-project="{esc(w)}">Archive</button></div>' for w in words[:3]) + '</div><dialog id="dialog"><h2>Confirm archive</h2><p id="subject"></p><button id="confirm">Archive project</button><button class="secondary" id="cancel">Keep project</button></dialog>'
        script = "let selected=null;const dialog=document.getElementById('dialog');document.querySelectorAll('[data-project]').forEach(b=>b.onclick=()=>{selected=b.dataset.project;document.getElementById('subject').textContent=selected;dialog.showModal();});document.getElementById('confirm').onclick=()=>{record({archived:selected});dialog.close();};document.getElementById('cancel').onclick=()=>dialog.close();"
        expected = {"archived": target}
    elif family == "tabs":
        goal = f"Open the Files tab and mark {target} as the favorite file."
        content = '<nav><button class="secondary" id="overview">Overview</button><button id="files">Files</button></nav><section id="home" class="panel">Workspace overview</section><section id="list" class="panel" hidden>' + ''.join(f'<div class="row"><span>{esc(w)}</span><button data-file="{esc(w)}">Favorite</button></div>' for w in words[:4]) + '</section>'
        script = "document.getElementById('files').onclick=()=>{document.getElementById('home').hidden=true;document.getElementById('list').hidden=false;};document.getElementById('overview').onclick=()=>{document.getElementById('home').hidden=false;document.getElementById('list').hidden=true;};document.querySelectorAll('[data-file]').forEach(b=>b.onclick=()=>record({favorite:b.dataset.file}));"
        expected = {"favorite": target}
    elif family == "counter":
        quantity = rng.randrange(2, 4) if split == "development" else rng.randrange(4, 6)
        goal = f"Set the quantity to {quantity} and save the order."
        content = '<section class="panel"><h2>Order quantity</h2><div class="count" id="count">1</div><div class="row"><button id="minus">Minus</button><button id="plus">Plus</button></div><button id="save">Save order</button></section>'
        script = "let count=1;function show(){document.getElementById('count').textContent=count;}document.getElementById('minus').onclick=()=>{count=Math.max(0,count-1);show();};document.getElementById('plus').onclick=()=>{count=Math.min(9,count+1);show();};document.getElementById('save').onclick=()=>record({quantity:count});"
        expected = {"quantity": quantity}
    elif family == "scroll":
        target = words[5]
        goal = f"Find the {target} destination farther down the page and choose it."
        content = '<p>Choose one destination. More are available below.</p>' + ''.join(f'<section class="card" style="min-height:145px"><h2>{esc(w)}</h2><button data-destination="{esc(w)}">Choose {esc(w)}</button></section>' for w in words[:7])
        script = "document.querySelectorAll('[data-destination]').forEach(b=>b.onclick=()=>record({destination:b.dataset.destination}));"
        expected = {"destination": target}
    else:
        goal = f"Search for {target} and open its result."
        content = '<form id="form" class="panel"><label>Search directory<input type="search" id="query"></label><button>Search</button></form><section id="results" class="panel" hidden></section>'
        script = "const entries="+js(words[:5])+";document.getElementById('form').onsubmit=e=>{e.preventDefault();const result=document.getElementById('results');result.replaceChildren();result.hidden=false;const term=document.getElementById('query').value.toLowerCase();for(const value of entries.filter(x=>x.toLowerCase().includes(term))){const b=document.createElement('button');b.textContent='Open '+value;b.onclick=()=>record({opened:value});result.appendChild(b);}if(!result.children.length)result.textContent='No matches';};"
        expected = {"opened": target}
    accent = rng.choice(("#275dc7", "#08776d", "#6e48aa"))
    return Case(f"screen-{split}-{index+1:02d}", split, family, layout, seed, goal,
                _page(family.title(), content, script, accent),
                json.dumps(expected, sort_keys=True, separators=(",", ":")))


def cases(split):
    if split not in SEEDS:
        raise ValueError("split must be development or verification")
    return tuple(_build(split, index) for index in range(8))


def demo_cases():
    """Explicit demonstrations reuse development fixtures; no verification score."""
    selected = cases("development")
    return selected[0], selected[3]


def check_receipt(case, actual):
    try:
        return type(actual) is dict and json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) == case.expected_json
    except (TypeError, ValueError):
        return False
