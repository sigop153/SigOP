# Symptom2Disease

Input: a patient's symptoms in one or two English sentences. Target: one of the 22 diseases
listed in objective.toml, spelled exactly. The executor answers inside
`[DIAGNOSIS]...[/DIAGNOSIS]`; a prediction is correct when the tagged diagnosis matches the
target after lower-casing and whitespace normalisation.

Several diseases share symptoms (dengue, malaria, typhoid; common cold, pneumonia).
