# Selected GT evaluation subset

Seed: `20260930`. Applied: `True`. Source GT rows: **138873**. Total selected: **50**.

The selector maximizes rare-label marginal coverage and includes image aspect/resolution diversity.
Every retained frame has at least one GT label; companion annotation files are filtered by image ID.
The JSON manifest stores a SHA-256 digest for every retained image.

| Dataset | Source rows | Selected | Unique selected GT labels |
|---|---:|---:|---:|
| openimages_common_214 | 57224 | 14 | 45 |
| openimages_rare_200 | 21991 | 12 | 42 |
| imagenet_multi | 50000 | 12 | 82 |
| hico | 9658 | 12 | 68 |

## Retained frames

| Dataset | Image | GT labels | Shape | Selection reason |
|---|---|---|---:|---|
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/b4bd159f5e631dc8.jpg` | teddy, picture frame, toy, birthday cake | 1024x768 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/c8a4b7bab0e293e5.jpg` | pet, rat, squirrel, hamster | 1024x680 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/96116d2b916ade74.jpg` | suite, mattress, bedroom, bed | 1024x680 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/8af609f92f94eba4.jpg` | christmas tree, winter, pine | 768x1024 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/bfa8dd6b7a851d39.jpg` | gym, treadmill, leg | 683x1024 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/1c11f389a2f2673a.jpg` | gas stove, microwave, washing machine | 1024x613 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/6d7433b4c9a6b24a.jpg` | restroom, faucet, toilet bowl | 1024x1024 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/bdcfc349991f0327.jpg` | street art, doodle, piano | 1024x640 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/e6bd64a60bbef623.jpg` | pumpkin, autumn, halloween | 1024x768 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/7c27063ed661f51c.jpg` | sandwich, vending machine, hamburger | 1024x576 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/2947f0ba0babf16b.jpg` | antelope, giraffe, ostrich | 1024x767 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/b1d3c1ff18a2cc76.jpg` | grape, blueberry, blackberry | 1024x765 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/fc547a967d92bc53.jpg` | duck, swan, flamingo | 1024x682 | rare-label and marginal-coverage greedy selection |
| openimages_common_214 | `datasets/openimages_common_214/imgs/test/48ad87161306a11e.jpg` | baboon, gorilla, monkey | 844x1024 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/585121ea3724ab63.jpg` | Personal water craft, Extreme sport, Boating, Surfing Equipment, Wind wave | 1024x614 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/0fd72579946be783.jpg` | Portrait photography, Digital camera, Close-up, Camera operator, Black-and-white | 1024x682 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/6bf13ba5d719c24a.jpg` | Auto racing, Touring car, Luxury vehicle, Performance car | 1024x681 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/01122addafb07701.jpg` | Vegetarian food, Frying, Junk food, Fried food | 1024x678 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/28493917194e1d24.jpg` | Extreme sport, Divemaster, Scuba diving, Underwater diving | 731x1024 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/2fc1fc08190a3217.jpg` | Crocodilia, Nile crocodile, Close-up, Scaled reptile | 1024x682 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/1cb4e916a7b86339.jpg` | Khinkali, Pelmeni, Jiaozi | 1024x1024 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/0c3240ecec541bfc.jpg` | Freight transport, Heavy cruiser, Frigate | 1024x683 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/6805657a52d30677.jpg` | Computer speaker, Headphones, Electronic instrument | 1024x683 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/1d72539d0364e219.jpg` | Rural area, Dog walking, Herding | 1024x681 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/63ba82a9b52b20da.jpg` | Horse racing, Equitation, Equestrianism | 1024x683 | rare-label and marginal-coverage greedy selection |
| openimages_rare_200 | `datasets/openimages_rare_200/imgs/test/32a70ecdbf132831.jpg` | Combat sport, Sumo, Grappling | 1024x768 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00012625.JPEG` | window shade, desk, monitor, couch, table lamp, pillow, television, quilt, wardrobe | 500x428 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00015184.JPEG` | microphone, accordion, acoustic guitar, electric guitar, folding chair, spotlight, stage, violin | 500x375 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00036160.JPEG` | computer mouse, joystick, notebook computer, laptop computer, desk, desktop computer, modem, music speaker, computer keyboard | 500x375 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00036499.JPEG` | wine bottle, wooden spoon, dining table, plate, guacamole, cabbage, red wine, restaurant | 500x375 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00006540.JPEG` | swim trunks / shorts, motorboat, sandbar, beach, paddle, bikini, one-piece bathing suit | 500x375 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00007145.JPEG` | water bottle, bottle cap, baby pacifier, perfume, tea cup, can opener, stopwatch | 500x375 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00029268.JPEG` | cornet, saxophone, oboe, trombone, steel drum, drum, drumstick | 500x375 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00034982.JPEG` | cowboy boot, Granny Smith apple, sunglasses, sunglasses, tennis ball, chain-link fence, messenger bag | 457x500 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00027015.JPEG` | manhole cover, traffic or street sign, sweatshirt, taxicab, plastic bag, jeans | 500x459 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00018817.JPEG` | broccoli, zucchini, cucumber, orange, lemon, banana | 500x375 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00000998.JPEG` | military uniform, tank, bulletproof vest, amphibious vehicle, assault rifle | 1075x1505 | rare-label and marginal-coverage greedy selection |
| imagenet_multi | `datasets/imagenet_multi/imgs/ILSVRC2012_val_00017903.JPEG` | sink, soap dispenser, sunscreen, hair spray, lotion | 357x500 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00004867.jpg` | person inspect oven, person operate oven, person carry pizza, person cook pizza, person hold pizza, person make pizza, person slide pizza | 640x480 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00002939.jpg` | person hold motorcycle, person push motorcycle, person race motorcycle, person ride motorcycle, person straddle motorcycle, person walk motorcycle | 640x570 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00003142.jpg` | person no_interaction bowl, person cut_with knife, person hold knife, person stick knife, person wield knife, person no_interaction sink | 640x480 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00007725.jpg` | person adjust skis, person inspect skis, person pick_up skis, person repair skis, person stand_on skis, person wear skis | 480x640 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00008248.jpg` | person board bus, person direct bus, person drive bus, person inspect bus, person ride bus, person sit_on bus | 640x438 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00000695.jpg` | person cut apple, person hold apple, person inspect apple, person peel apple, person pick apple, person wash apple | 640x480 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00000374.jpg` | person adjust tie, person hold tie, person inspect tie, person pull tie, person tie tie, person wear tie | 640x480 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00000255.jpg` | person drink_with cup, person hold cup, person pour cup, person sip cup, person smell cup | 640x425 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00002990.jpg` | person no_interaction bus, person drive car, person park car, person ride car, person no_interaction stop_sign | 640x427 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00001170.jpg` | person dry cat, person hold cat, person hug cat, person pet cat, person wash cat | 640x480 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00002595.jpg` | person hold sheep, person hug sheep, person pet sheep, person walk sheep, person wash sheep | 640x317 | rare-label and marginal-coverage greedy selection |
| hico | `datasets/hico/imgs/HICO_test2015_00003573.jpg` | person carry carrot, person cut carrot, person hold carrot, person peel carrot, person wash carrot | 640x480 | rare-label and marginal-coverage greedy selection |
