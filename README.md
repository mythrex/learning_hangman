# Learning Hangman

**Project Description:**

The "Learning Hangman" project is a fun and educational endeavor based on the classic word-guessing game. In this game, players are challenged to guess the correct word by suggesting letters, all while avoiding the dreaded hangman's noose.

**Game Inspiration:**

Our project draws inspiration from the online game [Hangaroo](https://www.play-games.com/game/4167/hangaroo.html), which serves as the basis for our implementation. In Hangaroo, players are presented with a word with missing letters and must guess the word by providing letters one at a time. Incorrect guesses result in the gradual drawing of a hangman figure.

**Learning Approach:**

What sets "Learning Hangman" apart is the use of state-of-the-art natural language processing techniques, specifically BERT (Bidirectional Encoder Representations from Transformers) with mask language modeling. By leveraging BERT, we've taught our system to not only play the game but also understand the underlying language patterns. The results have been quite promising, with the system demonstrating a strong ability to guess the correct words.

**Getting Started:**

To get started with "Learning Hangman," follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/learning-hangman.git
   ```

**Results**

```
┌────────────┬──────────┐
│ num tries  ┆ value    │
│ ---        ┆ ---      │
│ str        ┆ f64      │
╞════════════╪══════════╡
│ min        ┆ 1.0      │
│ max        ┆ 13.0     │
│ null_count ┆ 0.0      │
│ mean       ┆ 7.406    │
│ std        ┆ 2.008282 │
│ count      ┆ 1000.0   │
│ median     ┆ 7.0      │
└────────────┴──────────┘
```

On an average the model gets correct word in 7 tries. 

**Feedback:**

We appreciate your feedback! If you have suggestions, encounter issues, or simply want to share your experience with "Learning Hangman," please don't hesitate to open an issue or reach out to us.

**License:**

This project is licensed under the [MIT License](LICENSE).