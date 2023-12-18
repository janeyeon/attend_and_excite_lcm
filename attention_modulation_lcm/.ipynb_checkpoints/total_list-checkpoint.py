def return_temp(image_idx):
    total_input_temp = {
    "ancmach" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 2;
Number of people and objects of Group0: P 2; O 0;
Number of people and objects of Group1: P 1; O 0;
Description;
Global : a ornamental flower gardens and destroyed castle, covered with old dirt and moss, grass.;
Group0 : the two people are in the left side.;
Group1 : the Large Robot are in the right, middle side.;
Group0 bounding box; [ xmin 38 ymin 195 xmax 122 ymax 274 ];
Group1 bounding box; [ xmin 325 ymin 198 xmax 440 ymax 314 ];
Group0;
P0: a man is wearing red shirts, looking at robot;
P1: a man is wearing brown shirts, looking at robot;
Group1;
P0: an ancient ruins of a giant robot, made by huge rocks, covered with dust, moss;
    """,

    "castle" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 2;
Number of people and objects of Group0: P 2; O 0;
Number of people and objects of Group1: P 2; O 0;
Description;
Global : a realistic sculpted castle and stone walls in the rocky mountains.;
Group0 : a small group of two man are walking in left under side.;
Group1 : a small group of  two woman are walking in right under side.;
Group0 bounding box; [ xmin 135 ymin 269 xmax 243 ymax 308 ];
Group1 bounding box; [ xmin 353 ymin 269 xmax 446 ymax 351 ];
Group0;
P0: a small size man is walking;
P1: a small size man is walking;
Group1;
P0: a small size woman is walking;
P1: a small size woman is walking;
    """,
    "christ" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 2;
Number of people and objects of Group0: P 2; O 0;
Number of people and objects of Group1: P 1; O 0;
Description;
Global : Christmas village with crowd people, night, shiny, Christmas colors muted.;
Group0 : the two people are in the left side.;
Group1 : the person is in the right side with christmas tree.;
Group0 bounding box; [ xmin 0 ymin 326 xmax 248 ymax 478 ];
Group1 bounding box; [ xmin 280 ymin 346 xmax 480 ymax 564 ];
Group0;
P0: a girl is talking to the friend;
P1: a girl with red coat;
Group1;
P0: a man wearing coat;
    """,


    "futurecity" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 1;
Number of people and objects of Group0: P 1; O 1;
Description;
Global : some futuristic city and flying ships, in the style of spiritual landscape, meticulously detailed.;
Group0 : A person next to the futuristic car.;
Group0 bounding box; [ xmin 307 ymin 201 xmax 430 ymax 346 ];
Group0;
P0: a person with futuristic uniform and goggle;
O0: a futuristic car;
    """, 
    "gallib" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 1;
Number of people and objects of Group0: P 2; O 1;
Description;
Global : interior multi-story library of a huge luxury home inside with an astral galactic style, nebulas, stunning.;
Group0 : the two people and robot are in the center bottom side.;
Group0 bounding box; [ xmin 209 ymin 428 xmax 404 ymax 599 ];
Group0;
P0: a man is wearing wizard unifrom;
P1: a woman is wearing wizard unifrom;
O0: a small white cute robot;
    """, 
    "ironman" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 2;
Number of people and objects of Group0: P 2; O 1;
Number of people and objects of Group1: P 2; O 0;
Description;
Global : The Legend of Zelda landscape, four girls are dancing on the ground.;
Group0 : two girls are dancing in the left side.;
Group1 : two girls are dancing in the right side.;
Group0 bounding box; [ xmin 10 ymin 162 xmax 244 ymax 381 ];
Group1 bounding box; [ xmin 371 ymin 190 xmax 515 ymax 335 ];
Group0;
P0: a girl, beautiful game character, wearing a red dress;
P1: a girl, beautiful game character, wearing traditional dress;
O0: a backpack;
Group1;
P0: a girl, beautiful game character, wearing traditional dress;
P1: a girl, beautiful game character, wearing a yellow dress;
    """,

    "machbat" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 2;
Number of people and objects of Group0: P 2; O 0;
Number of people and objects of Group1: P 1; O 0;
Description;
Global : a Baroque-style battle scene with futuristic robots and a golden palace in the background.;
Group0 : the two people are in the left side.;
Group1 : the large robot are in the left, middle side.;
Group0 bounding box; [ xmin 456 ymin 236 xmax 583 ymax 330 ];
Group1 bounding box; [ xmin 1 ymin 118 xmax 266 ymax 345 ];
Group0;
P0: a man attacking robot;
P1: a man attacking robot;
Group1;
P0: a giant, large size Baroque-style robot attacking people;
    """,

    "robot" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 2;
Number of people and objects of Group0: P 1; O 0;
Number of people and objects of Group1: P 1; O 0;
Description;
Global : a Futuristic battle scene, a detroit city full of smokes, stone, fire in the background.;
Group0 : the Large Robot are in the left, middle side.;
Group1 : the person is in the right side.;
Group0 bounding box; [ xmin 0 ymin 240 xmax 302 ymax 463 ];
Group1 bounding box; [ xmin 331 ymin 323 xmax 480 ymax 446 ];
Group0;
P0: anatomically correct unspeakable unimaginable robot creature,gearwheel, clock parts, attacking people;
Group1;
P0: a man attacked by robot;
    """,

    "sea" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 1;
Number of people and objects of Group0: P 1; O 0;
Description;
Global : a Under the beautiful deep sea teeming with vibrant corals, colorful, vivid fishes.;
Group0 : A diver explores a breathtakingly in to the sea, center of the image.;
Group0 bounding box; [ xmin 185 ymin 151 xmax 269 ymax 255 ];
Group0;
P0: a Diver with skin scuber;
    """,

    "ship" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 1;
Number of people and objects of Group0: P 2; O 0;
Description;
Global : Photo of a ultra realistic sailing ship, dramatic light, big wave, pale sunrise, trending on artstation.;
Group0 : the person is watching sailor ship.;
Group0 bounding box; [ xmin 443 ymin 162 xmax 637 ymax 313 ];
Group0;
P0: a person with brown shirts;
P1: a person with white shirts;
    """,

    "snowforest" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 1;
Number of people and objects of Group0: P 2; O 1;
Description;
Global : a large big lake surrounded by a frozen taiga forest, snow on the trees, water, winter, midnight, full moon.;
Group0 : the two people are on the boat, center of the picture.;
Group0 bounding box; [ xmin 170 ymin 144 xmax 479 ymax 411 ];
Group0;
P0: a man is wearing fur clothes and riding a boat;
P1: a woman is wearing fur clothes and riding a boat;
O0: a large boat;
    """,

    "spcity" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 1;
Number of people and objects of Group0: P 2; O 0;
Description;
Global : steam punk city, gothic punk, rain, night, dim yellow light.;
Group0 : two people are walking down the street;
Group0 bounding box; [ xmin 100 ymin 220 xmax 304 ymax 451 ];
Group0;
P0: a person with a black coat;
P1: a person with an umbrella;
    """,


    "starwars" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 3;
Number of people and objects of Group0: P 3; O 0;
Number of people and objects of Group1: P 1; O 1;
Number of people and objects of Group2: P 4; O 0;
Description;
Global : an Alien planet, background is sparkling Milky way and lots of stars.;
Group0 : the Star Wars Characters on the Alien planet are in the left side.;
Group1 : the Darth Vader with a light saber in the middle and front side.;
Group2 : the Star Wars Characters on the Alien planet in the right side.;
Group0 bounding box; [ xmin 1 ymin 154 xmax 168 ymax 273 ];
Group1 bounding box; [ xmin 263 ymin 105 xmax 413 ymax 413 ];
Group2 bounding box; [ xmin 430 ymin 154 xmax 640 ymax 344 ];
Group0;
P0: a stormtrooper attacking people;
P1: a stormtrooper attacking people;
P2: a stormtrooper attacking people
P0: A Darth Vader, handsome, holding a lightsabor, highly detailed;
O0: a lightsaber;
Group2;
P0: a stormtrooper attacking people;
P1: a stormtrooper attacking peoples;
P2: a stormtrooper attacking people;
P3: a stormtrooper attacking people;
    """,

    "zelda" : \
    """
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 2;
Number of people and objects of Group0: P 2; O 1;
Number of people and objects of Group1: P 2; O 0;
Description;
Global : The Legend of Zelda landscape, four girls are dancing on the ground.;
Group0 : two girls are dancing in the left side.;
Group1 : two girls are dancing in the right side.;
Group0 bounding box; [ xmin 10 ymin 162 xmax 244 ymax 381 ];
Group1 bounding box; [ xmin 371 ymin 190 xmax 515 ymax 335 ];
Group0;
P0: a girl, beautiful game character, wearing a red dress;
P1: a girl, beautiful game character, wearing traditional dress;
O0: a backpack;
Group1;
P0: a girl, beautiful game character, wearing traditional dress;
P1: a girl, beautiful game character, wearing a yellow dress;
    """
    }

    #########################################################
        
    total_output_temp = {
    "ancmach" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 0;
P0: [ person a 79 209 b 70 217 c 63 218 d 56 232 e 59 247 f 77 216 g 81 230 h 87 245 i 65 248 j 65 271 k 63 271 l 74 246 m 74 271 n 73 271 o 77 207 p 79 207 q 72 208 r 79 207 ]; [ xmin 38 ymin 195 xmax 105 ymax 274 ];
P1: [ person a 104 211 b 102 217 c 95 217 d 93 227 e 97 234 f 109 217 g 112 227 h 116 234 i 98 236 j 98 251 k 98 266 l 107 235 m 106 251 n 106 266 o 102 209 p 106 209 q 99 209 r 107 209 ]; [ xmin 79 ymin 198 xmax 122 ymax 274 ];
Number of people and objects of Group1: P 1; O 0;
P0: [ person a 373 215 b 370 230 c 348 231 d 338 253 e 336 268 f 392 229 g 399 247 h 408 261 i 353 270 j 353 298 k 353 324 l 382 270 m 382 297 n 384 322 o 369 213 p 376 213 q 364 215 r 382 215 ]; [ xmin 325 ymin 198 xmax 440 ymax 314 ];
    """,

    "castle" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 0;
P0: [ person a 157 282 b 157 291 c 144 292 d 141 303 e 145 299 f 170 290 g 174 300 h 175 299 i 152 306 j 152 312 k 152 312 l 167 306 m 167 312 n 169 312 o 155 280 p 160 280 q 151 280 r 163 280 ]; [ xmin 135 ymin 269 xmax 179 ymax 308 ];
P1: [ person a 196 282 b 194 287 c 184 287 d 178 298 e 176 307 f 203 287 g 206 298 h 209 307 i 187 309 j 186 319 k 186 319 l 199 309 m 201 319 n 202 319 o 194 280 p 198 280 q 189 280 r 200 280 ]; [ xmin 165 ymin 269 xmax 243 ymax 308 ];
Number of people and objects of Group1: P 2; O 0;
P0: [ person a 375 286 b 370 293 c 361 293 d 354 304 e 362 310 f 379 293 g 383 304 h 379 314 i 364 314 j 364 334 k 364 352 l 376 315 m 378 334 n 379 351 o 374 285 p 376 285 q 370 285 r 377 286 ]; [ tr 386 284 378 284 ];
P1: [ person a 414 284 b 426 291 c 421 291 d 421 302 e 418 311 f 431 291 g 434 303 h 418 311 i 426 316 j 426 335 k 427 352 l 433 316 m 433 335 n 433 352 o 414 282 p 416 282 q 419 282 r 420 283 ]; [ tr 422 285 435 286 ];
    """,
    "christ" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 0;
P0: [ person a 105 356 b 82 378 c 99 378 d 116 407 e 112 432 f 64 378 g 57 407 h 63 433 i 87 432 j 101 462 k 95 491 l 63 432 m 67 461 n 65 491 o 103 352 p 105 352 q 100 354 r 75 352 ]; [ xmin 0 ymin 326 xmax 120 ymax 478 ];
P1: [ person a 213 378 b 195 398 c 218 398 d 234 433 e 236 459 f 172 398 g 162 433 h 169 459 i 213 462 j 216 497 k 213 526 l 181 462 m 184 497 n 185 526 o 211 375 p 213 375 q 206 378 r 187 375 ]; [ xmin 121 ymin 354 xmax 248 ymax 478 ];
Number of people and objects of Group1: P 1; O 0;
P0: [ person a 366 403 b 388 433 c 358 433 d 331 475 e 303 498 f 418 433 g 433 475 h 420 498 i 366 525 j 366 591 k 366 591 l 408 525 m 408 591 n 408 591 o 366 397 p 376 397 q 366 397 r 400 397 ]; [ xmin 280 ymin 346 xmax 480 ymax 564 ];
    """,

    "futurecity" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 1; O 1;
P0: [ person a 356 227 b 342 242 c 340 242 d 337 258 e 352 266 f 344 242 g 346 258 h 352 266 i 339 274 j 352 287 k 350 314 l 343 274 m 355 287 n 355 312 o 355 224 p 355 224 q 350 224 r 345 224 ]; [ xmin 307 ymin 211 xmax 389 ymax 346 ];
O0: [ xmin 321 ymin 212 xmax 518 ymax 382 ];
    """, 

    "gallib" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 1;
P0: [ person a 243 452 b 245 468 c 230 468 d 222 487 e 215 498 f 259 469 g 268 488 h 282 498 i 237 508 j 243 538 k 239 566 l 256 508 m 251 538 n 247 565 o 240 450 p 247 450 q 237 452 r 253 452 ]; [ xmin 209 ymin 428 xmax 303 ymax 599 ];
P1: [ person a 329 445 b 344 455 c 333 455 d 326 473 e 317 490 f 355 455 g 362 475 h 366 490 i 333 495 j 335 524 k 333 554 l 350 496 m 351 525 n 352 555 o 327 443 p 331 443 q 327 443 r 341 443 ]; [ xmin 303 ymin 429 xmax 375 ymax 599 ];
O0: [ xmin 336 ymin 474 xmax 385 ymax 583 ];
    """,

    "ironman" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 1;
P0: [ person a 48 193 b 40 218 c 15 222 d 11 261 e 34 255 f 65 215 g 77 255 h 73 255 i 33 298 j 29 353 k 29 395 l 65 294 m 63 353 n 63 395 o 42 187 p 54 187 q 29 189 r 58 189 ]; [ xmin 10 ymin 162 xmax 94 ymax 358 ];
P1: [ person a 194 222 b 181 244 c 152 246 d 145 282 e 170 277 f 210 242 g 221 282 h 207 309 i 161 321 j 165 373 k 168 390 l 193 319 m 193 373 n 195 390 o 188 215 p 198 215 q 174 215 r 202 215 ]; [ xmin 136 ymin 185 xmax 244 ymax 381 ];
O0: [ xmin 15 ymin 249 xmax 55 ymax 305 ];
Number of people and objects of Group1: P 2; O 0;
P0: [ person a 390 211 b 393 226 c 378 226 d 370 245 e 373 263 f 408 226 g 416 242 h 404 259 i 384 261 j 384 291 k 384 321 l 404 263 m 404 291 n 404 321 o 386 208 p 395 208 q 382 209 r 402 209 ]; [ xmin 371 ymin 190 xmax 424 ymax 335 ];
P1: [ person a 448 211 b 467 225 c 454 224 d 450 253 e 442 276 f 481 225 g 494 252 h 481 274 i 450 276 j 450 308 k 450 338 l 471 278 m 471 310 n 471 341 o 446 207 p 454 207 q 446 207 r 466 207 ]; [ xmin 430 ymin 192 xmax 515 ymax 329 ];
    """, 

    "machbat" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 0;
P0: [ person a 494 254 b 484 269 c 474 269 d 468 293 e 486 306 f 494 269 g 498 286 h 504 298 i 471 304 j 476 328 k 476 328 l 488 304 m 484 328 n 484 328 o 493 252 p 496 251 q 488 252 r 496 252 ]; [ xmin 456 ymin 236 xmax 510 ymax 318 ];
P1: [ person a 524 264 b 526 279 c 516 279 d 508 302 e 506 321 f 536 279 g 541 302 h 546 317 i 518 323 j 516 342 k 517 342 l 533 323 m 535 342 n 537 342 o 523 261 p 526 261 q 521 264 r 530 263 ]; [ xmin 498 ymin 246 xmax 583 ymax 330 ];
Number of people and objects of Group1: P 1; O 0;
P0: [ person a 115 170 b 110 193 c 90 194 d 70 240 e 65 275 f 130 192 g 137 229 h 163 256 i 110 275 j 115 334 k 121 375 l 137 271 m 150 328 n 156 375 o 108 163 p 118 163 q 100 163 r 127 163 ]; [ xmin 1 ymin 118 xmax 266 ymax 345 ]; d 450 253 e 442 276 f 481 225 g 494 252 h 481 274 i 450 276 j 450 308 k 450 338 l 471 278 m 471 310 n 471 341 o 446 207 p 454 207 q 446 207 r 466 207 ]; [ xmin 430 ymin 192 xmax 515 ymax 329 ];
    """,

    "robot" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 1; O 0;
P0: [ person a 245 322 b 233 332 c 199 331 d 174 380 e 186 424 f 266 333 g 285 377 h 282 416 i 201 438 j 197 474 k 193 474 l 241 441 m 245 474 n 245 474 o 236 313 p 250 316 q 223 307 r 258 316 ]; [ xmin 0 ymin 240 xmax 298 ymax 463 ];
Number of people and objects of Group1: P 1; O 0;
P0: [ person a 381 351 b 423 385 c 426 383 d 419 425 e 381 430 f 421 388 g 419 440 h 381 442 i 436 446 j 421 432 k 423 442 l 426 449 m 414 432 n 417 442 o 383 347 p 384 348 q 419 351 r 400 353 ]; [ xmin 331 ymin 323 xmax 480 ymax 446 ];
    """,

    "sea" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 1; O 0;
P0: [ person a 241 174 b 233 191 c 218 192 d 211 213 e 218 233 f 248 190 g 254 213 h 256 233 i 223 235 j 230 254 k 227 264 l 242 233 m 254 253 n 254 264 o 237 170 p 244 170 q 231 173 r 246 173 ]; [ xmin 185 ymin 151 xmax 269 ymax 255 ];
    """,

    "ship" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 0;
P0: [ person a 489 177 b 494 193 c 482 194 d 472 216 e 467 234 f 505 192 g 511 215 h 519 234 i 484 235 j 484 268 k 484 298 l 500 234 m 502 268 n 504 298 o 488 175 p 492 175 q 487 177 r 498 177 ]; [ xmin 443 ymin 162 xmax 534 ymax 313 ];
P1: [ person a 574 192 b 587 209 c 567 209 d 553 232 e 540 250 f 607 209 g 619 232 h 621 250 i 567 250 j 567 282 k 567 315 l 592 250 m 592 282 n 594 315 o 572 189 p 578 189 q 572 192 r 592 192 ]; [ xmin 509 ymin 173 xmax 637 ymax 313 ];
    """,

    "snowforest" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 1;
P0: [ person a 271 181 b 239 208 c 221 211 d 223 276 e 271 276 f 257 205 g 269 258 h 315 258 i 223 318 j 223 397 k 223 466 l 253 314 m 253 397 n 253 466 o 267 175 p 271 175 q 241 175 r 253 175 ]; [ xmin 170 ymin 144 xmax 369 ymax 411 ];
P1: [ person a 356 192 b 380 222 c 364 220 d 352 261 e 314 267 f 405 225 g 418 271 h 380 271 i 356 311 j 356 388 k 356 456 l 384 311 m 384 388 n 384 456 o 356 185 p 364 185 q 356 188 r 380 188 ]; [ xmin 282 ymin 153 xmax 479 ymax 411 ];
O0: [ xmin 168 ymin 134 xmax 480 ymax 410 ];
    """,

    "spcity" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 0;
P0: [ person a 202 267 b 185 291 c 189 293 d 189 322 e 204 315 f 182 290 g 182 316 h 202 313 i 195 344 j 202 378 k 209 418 l 188 341 m 195 377 n 202 415 o 202 263 p 202 263 q 197 265 r 190 265 ]; [ xmin 153 ymin 245 xmax 238 ymax 447 ];
P1: [ person a 133 243 b 124 257 c 118 257 d 116 277 e 130 276 f 131 257 g 136 274 h 133 290 i 122 299 j 122 335 k 122 368 l 130 298 m 130 335 n 130 368 o 131 240 p 134 240 q 125 241 r 135 241 ]; [ xmin 100 ymin 220 xmax 154 ymax 351 ];
    """,

    "starwars" : \
    """\
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
    """,

    "zelda" : \
    """\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 2; O 1;
P0: [ person a 48 193 b 40 218 c 15 222 d 11 261 e 34 255 f 65 215 g 77 255 h 73 255 i 33 298 j 29 353 k 29 395 l 65 294 m 63 353 n 63 395 o 42 187 p 54 187 q 29 189 r 58 189 ]; [ xmin 10 ymin 162 xmax 94 ymax 358 ];
P1: [ person a 194 222 b 181 244 c 152 246 d 145 282 e 170 277 f 210 242 g 221 282 h 207 309 i 161 321 j 165 373 k 168 390 l 193 319 m 193 373 n 195 390 o 188 215 p 198 215 q 174 215 r 202 215 ]; [ xmin 136 ymin 185 xmax 244 ymax 381 ];
O0: [ xmin 15 ymin 249 xmax 55 ymax 305 ];
Number of people and objects of Group1: P 2; O 0;
P0: [ person a 390 211 b 393 226 c 378 226 d 370 245 e 373 263 f 408 226 g 416 242 h 404 259 i 384 261 j 384 291 k 384 321 l 404 263 m 404 291 n 404 321 o 386 208 p 395 208 q 382 209 r 402 209 ]; [ xmin 371 ymin 190 xmax 424 ymax 335 ];
P1: [ person a 448 211 b 467 225 c 454 224 d 450 253 e 442 276 f 481 225 g 494 252 h 481 274 i 450 276 j 450 308 k 450 338 l 471 278 m 471 310 n 471 341 o 446 207 p 454 207 q 446 207 r 466 207 ]; [ xmin 430 ymin 192 xmax 515 ymax 329 ];
    """
    }

    return total_input_temp[image_idx], total_output_temp[image_idx]