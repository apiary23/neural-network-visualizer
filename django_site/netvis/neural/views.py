from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
from json import dumps
import logging

from network import Network
from network import data_from_csv

logger = logging.getLogger(__name__)

nn = []
t_d = []

def index(req):
    return render(req, 'neural.pug', context={})

def getNet(req):
    global nn, t_d
    t_d = data_from_csv("{}/network/data_banknote_authentication.csv"
                          .format(settings.BASE_DIR), 4)
    t_d.normalize();
    nn = Network([4,8,2])
    return HttpResponse(dumps({'status': 'ok'}), content_type='application/json')

def trainNet(req):
    global nn
    bs = int(req.GET.get("batchSize"))
    return HttpResponse(dumps(nn.train_alittle(t_d, batch_size=bs)), content_type='application/json')
