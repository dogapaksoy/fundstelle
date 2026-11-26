# Hand audit of 20 accepted rules

Random sample (seed 1) of rules the NLI check accepted, from both answer conditions,
compared with the statute text by hand.

| # | passage | verdict | note |
|---|---|---|---|
| 0 | BGB § 922 | simplified | "darf nicht ohne Zustimmung verändert werden" drops the condition "solange einer der Nachbarn ein Interesse hat" |
| 1 | StGB § 348 Abs. 1 | correct | |
| 2 | ZPO § 36 Abs. 1 | correct | one of the listed cases (Nr. 2) |
| 3 | BGB § 922 | correct | |
| 4 | BGB § 919 Abs. 1 | correct | |
| 5 | BGB § 1420 | correct | |
| 6 | BGB § 1605 Abs. 1 | simplified | drops "soweit zur Feststellung eines Unterhaltsanspruchs erforderlich" |
| 7 | BGB § 921 | correct | |
| 8 | BGB § 632 Abs. 2 | correct | verbatim |
| 9 | ZPO § 796 Abs. 3 | correct | verbatim |
| 10 | BGB § 637 Abs. 1 | **wrong** | "wenn der Unternehmer die Nacherfüllung verweigert": the statute says "wenn nicht der Unternehmer die Nacherfüllung zu Recht verweigert" and requires an expired deadline first. Entailment 0.99. |
| 11 | BGB § 93 | correct | |
| 12 | StGB § 273 Abs. 1 | simplified | "Die Veränderung von amtlichen Ausweisen ist strafbar" drops "zur Täuschung im Rechtsverkehr" |
| 13 | BGB § 1782 Abs. 1 | correct | |
| 14 | BGB § 2064 | correct | verbatim |
| 15 | BGB § 613a Abs. 5 | correct | shortened list |
| 16 | ZPO § 725 | correct | |
| 17 | BGB § 613a Abs. 4 | correct | |
| 18 | StGB § 226 Abs. 2 | correct | |
| 19 | BGB § 917 Abs. 1 | simplified | "bitten ... die Verbindung herzustellen" weakens "verlangen, dass sie die Benutzung dulden" |

15 correct, 4 simplified (a condition dropped), 1 wrong.
