To download the e-SNLI and MultiNLI datasets, please visit the corresponding dataset websites https://github.com/OanaMariaCamburu/e-SNLI and https://cims.nyu.edu/~sbowman/multinli/. The SICK dataset is included within the code.
The installation of the IBM transition parser can be found at https://github.com/IBM/transition-amr-parser.
The installation of the AMR to FOL converter can be found at https://github.com/chanind/amr-logic-converter.

The pipeline takes a pair consisting of a premise (and explanation) sentence and a claim sentence as input. It converts these sentences into propositional logic form via an AMR parser and an AMR-to-FOL converter. Then, the prove() function in the code classifies the propositional logic pair as entailment, contradiction, or neutral using relaxed methods.
