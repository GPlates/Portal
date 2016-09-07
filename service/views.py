from django.shortcuts import render, render_to_response
from django.template import RequestContext

def index(request):
    return render_to_response(
        'service.html',
        context_instance = RequestContext(request,
            {}))

def api_ref(request):
    return render_to_response(
        'api_ref.html',
        context_instance = RequestContext(request,
            {}))

def api_examples(request):
    viewname = request.GET.get('view','coastlines')
    return render_to_response(
        'd3_map.html',
        context_instance = RequestContext(request,
            {
                'view_name' : viewname
            }))

