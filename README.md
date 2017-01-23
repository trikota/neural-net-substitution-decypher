# neural-net-substitution-decypher
Recurrent net built with Keras.

## Basic idea ##

It uses information about all encyphered characters occurrences and the fact that cyphertext is written in English.
Input vector is fixed size. Each value x position p represents how far away character x is in a list of characters reverse-sorted by occurences in message. Get it? Here is an example.

String "hello" .
l-2, e-1, o-1, h-1 
l is on first place, then goes e, o, h .
Input for l would be "3,1,0,0,2" and desirable output "l".
For e - "2,0,25,25,1" (25 because we shift it, like it's a cyclic alphabet)
For o - "1,25,24,24,0"

There is clearly no difference between e, o and h in the example. In reality when two characters have same number of occurrences we have to decide randomly on which place to put them.
I hope you got the idea.
This vectorization method showed far better result than others I've tried.

BTW, because output is vector of probabilities for single char, some fancy algorithm can be implemented to find the best most-probable overall solution.

## Model ##

The model has 3 LSTM layers with 512 nodes. I'm sure there is better architecture, unfortunately I don't have any GPU to find it.
There is pretrained model on English wiki articles in this repo included.

## Examples ##
```
of the pediiine but was then exposed by jumie pallish when she disioveled his tl
of the pediiine but was then exposed by jumie pallish when she disioveled his tl

another supply of the medicine but was then exposed by julie parrish when she di
ahothes supply of the pedibihe but was theh exposed by julie passish wheh she di

protocol are often used to distribute icalendar data about an event and to publi
pnotolol ane often uued to dcutncbute clalendan data about an event and to publc
```
