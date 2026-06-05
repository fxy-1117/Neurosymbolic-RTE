"""Symbolic entailment/contradiction reasoner translated from the notebook.

The function bodies intentionally stay close to the notebook implementation so
that fixed-sample runs remain comparable. Runtime objects such as the neural
similarity scorer and spaCy model are injected by `configure_runtime()`.
"""

import copy
import inflect
import os
import re
from pathlib import Path

import nltk
import sympy
from amr_logic_converter import types
from pysat.formula import CNF
from pysat.solvers import Solver
from sympy import Symbol
from sympy.logic.boolalg import to_cnf


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_NLTK_DATA = _PROJECT_ROOT / "nltk_data"
_LOCAL_NLTK_DATA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NLTK_DATA", str(_LOCAL_NLTK_DATA))
if str(_LOCAL_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(_LOCAL_NLTK_DATA))

from nltk.tokenize import word_tokenize
from word_forms.word_forms import get_word_forms

nlp = None


def get_substring(s, w1, w2):
    """Return the sentence span connecting two AMR argument concepts."""
    p = inflect.engine()
    if w1.isnumeric():
        w1 = p.number_to_words(w1)
    if w2.isnumeric():
        w2 = p.number_to_words(w2)

    w11 = []
    if w1 == "be-located-at":
        w11 = ["on", "at", "in"]

    w22 = []
    if w2 == "be-located-at":
        w22 = ["on", "at", "in"]

    w111 = []
    if w1 == "person":
        w111 = [str(tok) for tok in nlp(s) if tok.dep_ == "nsubj"]

    w222 = []
    if w2 == "person":
        w222 = [str(tok) for tok in nlp(s) if tok.dep_ == "nsubj"]

    sub1 = [j for i in get_word_forms(w1, 0.7) for j in get_word_forms(w1, 0.7)[i]]
    sub1 += [w1] + w11 + w111
    sub2 = [j for i in get_word_forms(w2, 0.7) for j in get_word_forms(w2, 0.7)[i]]
    sub2 += [w2] + w22 + w222

    search1 = 0
    search2 = 999
    c1 = 0
    c2 = 0

    token = word_tokenize(s.lower())
    for i in range(len(token)):
        if token[i] in sub1:
            c1 = 1
            search1 = i
        elif token[i] in sub2:
            c2 = 1
            search2 = i
    if c1 == 0 or c2 == 0:
        return False

    if search1 > search2:
        return " ".join(token[search2 : search1 + 1])
    return " ".join(token[search1 : search2 + 1])


def combine(final, f=False):
    """Combine extracted AMR predicates into a SymPy boolean formula."""
    init = True
    for i in final:
        if type(i) == list:
            tem = True

            tem = tem & combine(i)
            if ~tem == -1:
                init = init & True
            elif ~tem == -2:
                if not f:
                    init = init & False
                else:
                    init = init & True
            else:
                init = init & ~tem
        else:
            init = init & i

    return init


def transform(formula, Var, X):
    """Replace AMR predicate names with propositional symbols."""
    final = copy.deepcopy(formula)
    for i in range(len(final)):
        if type(final[i]) == list:
            if final[i][0] == "ARG":
                key = " ".join([Var[final[i][1]], Var[final[i][2]], final[i][3]])
                if key not in X:
                    continue
                final[i] = X[key]
            else:
                final[i] = transform(final[i], Var, X)

        else:
            if final[i] not in X:
                continue
            final[i] = X[final[i]]

    return final


def extract(formula):
    """Extract monadic predicates, dyadic ARG predicates, and variables."""
    and_list = []
    var = {}
    arg = []
    if type(formula) == types.Not:
        nested_and, nested_var, nested_arg = extract(formula.body)
        return [nested_and], {**nested_var, **var}, arg + nested_arg

    for i in formula.args:
        if type(i) == types.Not:
            nested_and, nested_var, nested_arg = extract(i.body)
            and_list.append(nested_and)
            var = {**nested_var, **var}
            arg = arg + nested_arg

        else:
            if i.predicate.symbol[0] == ":":
                terms = [i.terms[j].value for j in range(0, len(i.terms))]
                and_list.append(["ARG"] + terms + [i.predicate.symbol])
                arg.append(terms + [i.predicate.symbol])
            else:
                predicate = re.sub(r"\-*[0-9]", "", i.predicate.symbol)
                and_list.append(predicate)
                var[i.terms[0].value] = predicate
    return and_list, var, arg


def score(s1, s2):
    """Placeholder replaced by configure_runtime()."""
    raise RuntimeError("reasoner.configure_runtime(scorer=...) must be called before prove().")


def pysat_formula(formula):
    """Convert a SymPy CNF formula into PySAT's list-of-clauses format."""
    tem_list = []
    for i in str(formula).split(" & "):
        if i[0] == "x":
            tem_list.append([int(i[1:])])
        else:
            tem_tem = []
            for j in i.replace("(", "").replace(")", "").split(" | "):
                if j[0] == "~":
                    tem_tem.append(int(j[2:]) * -1)
                elif j[0] == "x":
                    tem_tem.append(int(j[1:]))
            tem_list.append(tem_tem)
    return tem_list


def substitute_by_similarity(x, y, replaceX, replaceXX, maxx, i, j, thre):
    """Map a claim predicate to the most similar premise predicate above threshold."""
    tems = score(x, y)

    if tems >= thre:
        if tems > maxx[i]:
            maxx[i] = tems
            replaceXX[i] = replaceX[j]
            return True
    return False


def prove(data, sent):
    """Classify one premise/claim pair as ent, con, neu, or both.

    `data` contains AMR-to-logic formulas for premise, claim, and optionally an
    explanation. `sent` is the original notebook row used for substring matching.
    """
    # Premise predicates are the reference symbol table. Claim predicates are
    # mapped into this table by exact string match or neural similarity.
    checkArg0 = []
    checkVaribale0 = {}
    for0,checkVaribale0,checkArg0 = extract(data[0])
    for i in checkArg0:
        for j in range(len(i)):
            if i[j] in checkVaribale0:
                i[j] = checkVaribale0[i[j]]
            else:
                checkVaribale0[i[j]] = i[j]

    replaceX = {}
    n = 1
    for i in checkVaribale0:
        if checkVaribale0[i][0] == ":":
            continue
        if checkVaribale0[i] not in replaceX:
            replaceX[checkVaribale0[i]] = Symbol('x'+str(n))  
            n+=1
    for i in checkArg0:
        if " ".join(i) not in replaceX:
            replaceX[" ".join(i)] = Symbol('x'+str(n))  
            n+=1

    # e-SNLI uses the human explanation as additional premise information.
    explanation = False
    if len(data)>2:
        explanation = True
    if explanation:
        checkArg2 = []
        checkVaribale2 ={}
        for2,checkVaribale2,checkArg2 = extract(data[2])
        for i in checkArg2:
            for j in range(len(i)):
                if i[j] in checkVaribale2:
                    i[j] = checkVaribale2[i[j]]
                else:
                    checkVaribale2[i[j]] = i[j]

        for i in checkVaribale2:
            if checkVaribale2[i][0] == ":":
                continue
            if checkVaribale2[i] not in replaceX:
                replaceX[checkVaribale2[i]] = Symbol('x'+str(n))  
                n+=1
        for i in checkArg2:
            if " ".join(i) not in replaceX:
                replaceX[" ".join(i)] = Symbol('x'+str(n))  
                n+=1
    checkArg11 = []
    checkVaribale11 ={}
    for1,checkVaribale11,checkArg11 = extract(data[1])
    
    quant = []
    for i in checkArg11:
        if i[-1][:6] == ":quant":
            try:
                quant.append(checkVaribale11[i[1]])
            except:
                continue
    replaceXX = {}

    thre = MATCH_THRESHOLD
    for i in checkArg11:
        for j in range(len(i)):
            if i[j] in checkVaribale11:
                i[j] = checkVaribale11[i[j]]
            else:
                checkVaribale11[i[j]] = i[j]

    maxx = {}
    for i in checkArg11:
        maxx[" ".join(i)] = 0
    for i in checkVaribale11:
        maxx[checkVaribale11[i]] = 0

    # Match claim unary predicates first, then relation predicates. Quantifier
    # predicates are initialized to True before the notebook's normal matching.
    for i in checkVaribale11:
        claim_predicate = checkVaribale11[i]
        if claim_predicate in quant:
            replaceXX[claim_predicate] = True
        if claim_predicate[0] == ":":
            continue
        if claim_predicate in replaceX:
            replaceXX[claim_predicate] = replaceX[claim_predicate]
        else:
            for j in replaceX:
                if len(j.split()) > 1:
                    continue
                substitute_by_similarity(
                    claim_predicate,
                    j,
                    replaceX,
                    replaceXX,
                    maxx,
                    claim_predicate,
                    j,
                    thre,
                )

            if maxx[claim_predicate] == 0:
                replaceXX[claim_predicate] = Symbol("x" + str(n))
                n += 1

    for i in checkArg11:
        relation_key = " ".join(i)
        if relation_key in replaceXX:
            continue
        if relation_key in replaceX:
            replaceXX[relation_key] = replaceX[relation_key]
        else:
            for j in replaceX:
                if len(j.split()) < 3:
                    substitute_by_similarity(
                        j,
                        " ".join([i[0], i[-2]]),
                        replaceX,
                        replaceXX,
                        maxx,
                        relation_key,
                        j,
                        thre,
                    )
                else:
                    tems3 = False
                    tems1 = get_substring(sent[1], i[0], i[-2])
                    tems2 = get_substring(sent[0], j.split()[0], j.split()[-2])
                    if explanation:
                        tems3 = get_substring(sent[2], j.split()[0], j.split()[-2])
                    if not tems1:
                        if tems2:
                            substitute_by_similarity(
                                " ".join([i[0], i[-2]]),
                                tems2,
                                replaceX,
                                replaceXX,
                                maxx,
                                relation_key,
                                j,
                                thre,
                            )

                        if tems3:
                            substitute_by_similarity(
                                " ".join([i[0], i[-2]]),
                                tems3,
                                replaceX,
                                replaceXX,
                                maxx,
                                relation_key,
                                j,
                                thre,
                            )

                        substitute_by_similarity(
                            " ".join([i[0], i[-2]]),
                            " ".join([j.split()[0], j.split()[1]]),
                            replaceX,
                            replaceXX,
                            maxx,
                            relation_key,
                            j,
                            thre,
                        )

                    if tems2 and tems1:
                        substitute_by_similarity(
                            tems1,
                            tems2,
                            replaceX,
                            replaceXX,
                            maxx,
                            relation_key,
                            j,
                            thre,
                        )

                    if tems3 and tems1:
                        substitute_by_similarity(
                            tems1,
                            tems3,
                            replaceX,
                            replaceXX,
                            maxx,
                            relation_key,
                            j,
                            thre,
                        )

            if maxx[relation_key] == 0:
                replaceXX[relation_key] = Symbol("x" + str(n))
                n += 1
            else:
                if i[0] in replaceXX:
                    replaceXX[i[0]] = True
                if i[-2] in replaceXX:
                    replaceXX[i[-2]] = True
    new_rex = {}
    for i in replaceXX:
        if i == "and":
            new_rex[i] = True
        if i.split()[0] == "and" and i.split()[-1][:3] == ":op":
            new_rex[i] = new_rex[i.split()[1]]
        else:
            new_rex[i] = replaceXX[i]

    new_re = {}
    for i in replaceX:
        tcc = 0
        for j in new_rex:
            if isinstance(new_rex[j], sympy.Not):
                if ~new_rex[j] == replaceX[i]:
                    new_re[i] = replaceX[i]
                    tcc = 1
            else:
                if new_rex[j] == replaceX[i]:
                    new_re[i] = replaceX[i]
                    tcc = 1
        if tcc == 0:
            new_re[i] = True

    formula0 = combine(transform(for0, checkVaribale0, replaceX))

    formula11 = combine(transform(for1, checkVaribale11, new_rex))
    if formula11 == -1:
        formula11 = True
    elif formula11 == -2:
        formula11 = False
    elif formula11 == 0:
        formula11 = False
    elif formula11 == 1:
        formula11 = True

    if explanation:
        formula2 = combine(transform(for2, checkVaribale2, replaceX))
        final_formula = to_cnf((formula0 & formula2) & ~(formula11))
        # Without forgetting, the contradiction check keeps every premise and
        # explanation atom in phi' before testing phi' & psi'.
        no_forgetting_contradiction = to_cnf(formula0 & formula2 & formula11)
    else:
        final_formula = to_cnf(formula0 & ~(formula11))
        no_forgetting_contradiction = to_cnf(formula0 & formula11)
    cnf = CNF(from_clauses=pysat_formula(final_formula))

    formula00 = combine(transform(for0, checkVaribale0, new_re))

    if explanation:
        formula22 = combine(transform(for2, checkVaribale2, new_re))
        if formula22 == -1:
            formula22 = True
        elif formula22 == -2:
            formula22 = False
        elif formula22 == 1:
            formula22 = True
        elif formula22 == 0:
            formula22 = False
        if formula00 == -1:
            formula00 = True
        elif formula00 == -2:
            formula00 = False
        elif formula00 == 1:
            formula00 = True
        elif formula00 == 0:
            formula00 = False
        # With forgetting, unmatched premise/explanation atoms are replaced
        # with True before the contradiction consistency check.
        final_formula11 = to_cnf(formula00 & formula22 & formula11)
    else:
        if formula00 == -1:
            formula00 = True
        elif formula00 == -2:
            # Notebook parity: the original no-explanation branch assigns this
            # misspelled name, so formula00 remains unchanged in that case.
            fomula00 = False

        final_formula11 = to_cnf(formula00 & formula11)

    contradiction_formula = final_formula11 if USE_FORGETTING else no_forgetting_contradiction
    cnf11 = CNF(from_clauses=pysat_formula(contradiction_formula))

    with Solver(name="Minisat22", bootstrap_with=cnf) as solver:
        check_ent = solver.solve()

    with Solver(name="Minisat22", bootstrap_with=cnf11) as solver:
        check_con1 = solver.solve()

    # PySAT returns True for satisfiable formulas and False for inconsistency.
    if not check_ent and check_con1:
        return "ent"
    elif not check_con1 and check_ent:
        return "con"
    elif not check_con1 and not check_ent:
        return "both"
    return "neu"


MATCH_THRESHOLD = 0.55
USE_FORGETTING = True


def configure_runtime(scorer=None, spacy_model=None, threshold=0.55, use_forgetting=True):
    """Install runtime objects used by the notebook-derived functions."""
    global score, nlp, MATCH_THRESHOLD, USE_FORGETTING
    if scorer is not None:
        score = scorer
    if spacy_model is not None:
        nlp = spacy_model
    MATCH_THRESHOLD = threshold
    USE_FORGETTING = use_forgetting


def set_threshold(threshold):
    """Set the neural predicate matching threshold used by prove()."""
    global MATCH_THRESHOLD
    MATCH_THRESHOLD = threshold


def set_use_forgetting(use_forgetting):
    """Select the contradiction formula used by prove()."""
    global USE_FORGETTING
    USE_FORGETTING = use_forgetting
