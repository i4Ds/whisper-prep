### Source: https://swissnlp.org/home/activities/datasets/

```
clip_id: unique clip identifier
clip_path: path to Swiss German clip in clip tar / folder
sentence: Standard German sentence
clip_created_at: creation date of clip
clip_is_valid: True (clip was validated as a correct Swiss German representation of the Standard German sentence by two or more users, see clip_n_votes_correct), False (clip was validated as NOT a correct Swiss German representation of the Standard German sentence by two or more users, see clip_n_votes_false) or empty string (not enough votes to decide)
sentence_id: unique sentence identifier
sentence_source: source of the sentence, tamedia_sentences = Swiss newspapers, cv_sentences = German Common Voice texts
client_id: unique speaker identifier (Warning: the same person may have gotten multiple ids when using the webapp on different occasions without registering. Also, the same person may have gotten multiple ids when using the webapp as an unregistered as well as a registered user.)
zipcode: zipcode of the origin municipality of a user's dialect
canton: canton of origin of a user's dialect
user_mean_clip_quality: mean quality or acceptance ratio of the speaker's clips in the interval [0, 1]
clip_n_votes_correct: number of users voting this clip as correct
clip_n_votes_false: number of users voting this clip as false
clip_n_times_reported: number of times this clip was reported by a user (reporting reasons: offensive language, grammar errors, wrong language, other)
sentence_n_times_reported: number of times this sentence was reported by a user(reporting reasons: offensive language, grammar errors, wrong language, too difficult to record, other)
age: speaker's age bracket
gender: speaker's gender
```
