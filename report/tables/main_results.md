| Input    | Model        |   Accuracy |   Macro-F1 | Best epoch   | Params   |
|:---------|:-------------|-----------:|-----------:|:-------------|:---------|
| input    | CNN          |     0.1652 |     0.1522 | 4            | 2020998  |
| input    | LSTM         |     0.177  |     0.1763 | 11           | 2120198  |
| output   | CNN          |     0.8333 |     0.8333 | 15           | 2020998  |
| output   | LSTM         |     0.7    |     0.6999 | 8            | 2120198  |
| combined | CNN          |     0.8274 |     0.8265 | 10           | 2020998  |
| combined | LSTM         |     0.6111 |     0.6104 | 14           | 2120198  |
| input    | TFIDF+LOGREG |     0.1815 |     0.1812 | n/a          | n/a      |
| output   | TFIDF+LOGREG |     0.8015 |     0.8011 | n/a          | n/a      |
| combined | TFIDF+LOGREG |     0.7748 |     0.7739 | n/a          | n/a      |