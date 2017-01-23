# neural-net-substitution-decypher
Recurrent net built with Keras.

It uses information about all encyphered characters occurrences and the fact that cyphertext is written in English.
Input vector is fixed size. Each value x position p represent how far away character x is in a list of characters reverse-sorted by occurences. Get it? Here is example.

String "hello" .
| l   |h    | o   |   e | 
| --- | --- | --- | --- |
| 2   | 1   |    1|    1|
