    input_temp = ["" , \
    '''
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 3;
Number of people and objects of Group0: P 3; O 0;
Number of people and objects of Group1: P 1; O 1;
Number of people and objects of Group2: P 4; O 0;n
Description;
Global : an Star Wars Characters on the Alien planet, background is sparkling Milky way and lots of stars.;
Group0 : the Star Wars Characters on the Alien planet are in the left side.;
Group1 : the Luke Skywalker with a light saber in the middle and front side.;
Group2 : the Star Wars Characters on the Alien planet in the right side.;
Group0 bounding box; [ xmin 1 ymin 154 xmax 168 ymax 273 ];
Group1 bounding box; [ xmin 263 ymin 105 xmax 413 ymax 413 ];
Group2 bounding box; [ xmin 430 ymin 154 xmax 640 ymax 344 ];
Group0;
P0: a stormtrooper attacking people;
P1: a stormtrooper attacking people;
P2: a stormtrooper attacking people;
Group1;
P0: A Luke Skywalker, handsome, holding a lightsabor, highly detailed;
O0: a lightsaber;
Group2;
P0: a stormtrooper attacking people;
P1: a stormtrooper attacking people;
P2: a stormtrooper attacking people;
P3: a stormtrooper attacking people;
''']
    
    output_temp = ["", \
    '''
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 3; O 0;
P0: [ person a 45 173 b 32 185 c 15 189 d 3 215 e 15 224 f 49 181 g 52 206 h 45 218 i 20 224 j 20 248 k 20 268 l 41 222 m 45 248 n 48 269 o 42 171 p 47 171 q 34 171 r 47 171 ]; [ xmin 1 ymin 154 xmax 61 ymax 269 ];
P1: [ person a 75 179 b 69 194 c 58 194 d 50 216 e 58 232 f 79 194 g 84 218 h 92 235 i 63 235 j 63 261 k 60 277 l 77 234 m 82 261 n 85 278 o 72 176 p 77 176 q 66 177 r 78 177 ]; [ xmin 42 ymin 160 xmax 103 ymax 273 ];
P2: [ person a 126 170 b 143 183 c 155 183 d 157 206 e 134 211 f 131 183 g 121 206 h 115 216 i 152 222 j 155 252 k 155 275 l 136 223 m 129 252 n 125 275 o 126 168 p 127 168 q 144 170 r 132 170 ]; [ xmin 102 ymin 154 xmax 168 ymax 273 ];
Number of people and objects of Group1: P 1; O 1;
P0: [ person a 323 141 b 338 183 c 307 185 d 276 221 e 265 192 f 369 182 g 388 231 h 377 267 i 321 284 j 291 332 k 295 391 l 362 284 m 381 337 n 395 400 o 321 136 p 333 133 q 321 138 r 346 133 ]; [ xmin 263 ymin 105 xmax 413 ymax 413 ];
O0: [ xmin 307 ymin 160 xmax 375 ymax 277 ];
Number of people and objects of Group0: P 4; O 0;
P0: [ person a 450 164 b 450 176 c 436 176 d 432 187 e 432 198 f 465 176 g 467 188 h 465 199 i 440 205 j 440 231 k 439 261 l 457 205 m 459 231 n 460 261 o 448 163 p 453 163 q 446 163 r 457 163 ]; [ xmin 430 ymin 154 xmax 466 ymax 308 ];
P1: [ person a 497 166 b 505 175 c 493 175 d 491 186 e 489 197 f 518 175 g 522 188 h 522 199 i 499 201 j 499 221 k 499 238 l 514 200 m 514 221 n 514 238 o 497 164 p 502 164 q 497 164 r 507 164 ]; [ xmin 488 ymin 158 xmax 518 ymax 300 ];
P2: [ person a 549 169 b 549 181 c 540 181 d 535 193 e 533 198 f 558 181 g 561 193 h 564 199 i 542 207 j 542 227 k 542 245 l 556 207 m 556 227 n 556 245 o 547 168 p 551 168 q 544 169 r 554 169 ]; [ xmin 528 ymin 161 xmax 561 ymax 304 ];
P3: [ person a 631 178 b 620 190 c 605 190 d 601 204 e 601 215 f 635 190 g 638 205 h 638 218 i 605 221 j 605 245 k 605 270 l 622 221 m 622 245 n 622 270 o 628 177 p 632 177 q 622 178 r 634 178 ]; [ xmin 582 ymin 165 xmax 640 ymax 344 ];
''']