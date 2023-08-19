
import time
import array

from digits_data import digits


def emlearn_create(emltrees):
    model = emltrees.new(10, 3000)

    # Load a CSV file with the model
    with open('eml_digits.csv', 'r') as f:
        emltrees.load_model(model, f)
    return model

def argmax(l):
    max_value = l[0]
    max_idx = 0
    for i, v in enumerate(l):
        if v > max_value:
            max_value = v
            max_idx = i
    return max_idx


def everywhere_run(clf, data):
    errors = 0
    for idx, x in enumerate(data):
        #x = array.array('f', x)
        out = clf.predict(x)
        if (idx != out):
            errors += 1
    return errors


def emlearn_run(model, data):
    errors = 0
    for idx, x in enumerate(data):
        out = model.predict(x)
        if (idx != out):
            errors += 1
    return errors

def m2c_run(m2c_digits, data):
    errors = 0
    for idx, x in enumerate(data):
        #x = array.array('f', x)
        scores = m2c_digits.score(x)
        out = argmax(scores)
        if (idx != out):
            errors += 1
    return errors

def none_run(data):
    errors = 0
    for idx, x in enumerate(data):
        #tmp = len(x)
        out = idx
        if (idx != out):
            errors += 1
    return errors

def benchmark():

    import gc

    #data = digits
    data = [ array.array('h', x) for x in digits ]


    print('model,errors,time_us')

    gc.collect()
    before = time.ticks_us()
    none_errors = none_run(data)
    gc.collect()
    after = time.ticks_us()
    none_duration = time.ticks_diff(after, before)
    print('none,{},{}'.format(none_errors, none_duration))

    import emltrees
    model = emlearn_create(emltrees)
    gc.collect()
    before = time.ticks_us()
    eml_errors = emlearn_run(model, data)
    gc.collect()
    after = time.ticks_us()
    eml_duration = time.ticks_diff(after, before)
    print('emlearn,{},{}'.format(eml_errors, eml_duration))
    del model
    del emltrees

    import everywhere_digits
    clf = everywhere_digits.RandomForestClassifier()  
    gc.collect()
    before = time.ticks_us()
    everywhere_errors = everywhere_run(clf, data)
    gc.collect()
    after = time.ticks_us()
    everywhere_duration = time.ticks_diff(after, before)
    print('everywhere,{},{}'.format(everywhere_errors, everywhere_duration))
    del clf
    del everywhere_digits

    #import m2c_digits
    #gc.collect()
    #before = time.ticks_us()
    #m2c_errors = m2c_run(m2c_digits, data)
    #gc.collect()
    #after = time.ticks_us()
    #m2c_duration = time.ticks_diff(after, before)
    #print('m2cgen,{},{}'.format(m2c_errors, m2c_duration))
    #del m2c_digits

