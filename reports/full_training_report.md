# Full Training Log Report

        This report records 100 deterministic training-log rows for the tiny educational
        transformer components. The numbers are synthetic but structured like a real
        training artifact: step, loss, attention entropy, and learning rate.

        ## First Five Rows

        | step | loss | attention_entropy | lr |
| --- | --- | --- | --- |
| 1 | 2.5307 | 1.9324 | 0.001 |
| 2 | 2.4699 | 1.9142 | 0.001 |
| 3 | 2.4048 | 1.8971 | 0.001 |
| 4 | 2.3368 | 1.8796 | 0.001 |
| 5 | 2.2769 | 1.8643 | 0.001 |

        ## Last Five Rows

        | step | loss | attention_entropy | lr |
| --- | --- | --- | --- |
| 96 | 0.42 | 0.8095 | 0.001 |
| 97 | 0.42 | 0.8041 | 0.001 |
| 98 | 0.42 | 0.7981 | 0.001 |
| 99 | 0.42 | 0.7911 | 0.001 |
| 100 | 0.42 | 0.7816 | 0.001 |

        ## Interpretation

        The project is not claiming production model training. It is a didactic
        from-scratch implementation with realistic experiment logging conventions.
