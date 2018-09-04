from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.

def index(req):
    return HttpResponse("fudge")

def chicken(req):
    return HttpResponse(chicken_svg)

def yay_pug(req):
    return render(req, 'foo.pug', context={'foo': 1})

chicken_svg = '''<svg width="640" height="480" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg">
 <!-- Created with SVG-edit - http://svg-edit.googlecode.com/ -->

 <g> <title>Layer 1</title> <path id="svg_3"
  d="m352.46851,423.3692c-11.05399,-4.73343 -23.08508,2.18457
  -34.89532,-4.70819c9.55139,-0.59464 23.90295,-0.04422
  21.36694,-14.20847c4.17947,-16.5405 -4.30148,-33.89258
  2.75931,-48.28815c-11.0567,-10.54922 -22.66284,-23.03906
  -28.55838,-37.77557c-1.1824,5.9451 -2.27374,10.67322
  -3.95544,0.42386c-0.96326,-5.01373 -0.37909,12.39914
  -4.06494,0.18033c-3.8869,-10.86469 -14.35159,-39.19339
  -17.59875,-12.91656c-1.34943,13.55872 -5.57977,3.93884
  -7.5141,-4.66656c-1.96649,-16.569 -9.06421,-31.91812
  -10.92328,-48.45406c7.26563,-10.47665 -1.03595,-11.90204
  -4.89789,-1.45078c-4.25385,16.57832 -7.89664,32.9864
  -3.46112,50.0666c-5.89551,4.8241 -17.27573,-22.4462
  -12.04124,-32.90555c4.54015,-11.83469 3.53185,-23.91704
  4.28674,-31.92183c4.15239,-9.03369 -4.04002,-18.84644
  0.31483,-28.47873c-11.12747,12.13144 -21.07867,20.52063
  -17.95175,39.55255c-1.38211,4.74893 4.63548,32.0479
  0.80225,21.99648c-7.23701,-19.62395 -7.2422,-41.35884
  -1.79672,-61.38197c0.00783,-15.28415 -12.38844,26.39291
  -11.47365,5.96399c-1.17278,-15.40611 5.17725,-32.34801
  11.18892,-44.52895c-6.84511,-8.14656 -29.02052,-2.37318
  -18.53085,-19.66959c10.32524,-8.79092 9.21915,-20.87808
  -4.83278,-10.24927c-21.69775,15.9361 -39.69972,36.40919
  -59.59029,54.50829c19.11087,-24.56604 40.65518,-48.33995
  67.31992,-64.82033c25.8546,-16.27974 63.34283,-17.96326
  85.65237,5.36841c5.67426,9.99481 18.86423,16.61989
  22.00751,25.61504c-2.31238,13.48659 6.39609,20.17328
  7.33084,32.02527c5.49408,9.1465 -4.95776,25.58875
  10.67911,20.8495c17.78564,-0.02528 42.12622,-3.96559
  43.51712,-26.14882c7.09799,-31.9485 10.36588,-67.64272
  33.09509,-93.06889c12.63275,-7.38432 4.78531,-5.41956
  -4.4451,-6.63544c-8.16144,-9.4978 8.23703,-24.74596
  16.31091,-26.48298c-1.39328,-22.44706 13.19427,13.45737
  17.36407,-3.02303c6.25687,0.11836 9.30661,15.1295
  13.11148,2.87762c11.3421,9.59721 23.33157,24.11761
  8.34738,37.48303c17.04584,8.1721 -17.36938,2.88054
  -2.68716,15.40137c12.11755,11.71254 3.5528,33.34698
  -6.39612,42.77829c-0.38574,7.0883 11.66501,17.38844
  14.08585,25.98943c12.51736,24.23186 15.02817,53.0603
  10.23004,79.61409c-6.75351,25.02634 -31.89313,35.9173
  -49.66757,51.81723c-15.18298,12.44659 -9.84213,34.28
  -17.03094,50.62726c0.71484,9.43826 5.98318,20.1875
  13.47037,25.88663c10.84714,6.06165 23.55756,9.97559
  32.32922,18.85181c-8.01086,4.1839 -24.34686,-12.79788
  -24.50354,0.48737c-11.90204,-15.45862 -29.76834,-5.08466
  -42.84912,-16.97375c10.88785,2.07349 25.03668,1.00558
  13.11548,-12.03717c-4.70636,-11.03079 -18.88483,-14.83716
  -19.25937,-28.16895c-13.35977,8.38748 -22.44818,19.80579
  -32.14923,31.19009c-16.10416,-1.00543 -16.79736,31.45789
  0.591,30.97571c7.13937,1.82541 33.94168,11.77591
  12.83603,9.20544c-6.29126,-1.12354 -13.11545,-5.06036
  -12.92258,2.32541c-2.88547,-0.51749 -5.45941,-1.95255
  -8.11554,-3.0975z" stroke-width="5" stroke="#a53b3b"
  fill="#ffaaaa"/> </g> </svg>'''