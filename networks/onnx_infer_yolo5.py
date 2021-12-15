import numpy as np
import cv2
import onnxruntime
import os


def prepare_image(path: str):
  image = cv2.imread(path)
  shape = image.shape[:2]
  new_shape = (1984,1984)
  color=(114, 114, 114)

  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  r = min(r, 1.0)

  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = (new_shape[1] - new_unpad[0])/2, (new_shape[0] - new_unpad[1])/2  # wh padding
  if shape[::-1] != new_unpad:  # resize
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
  image = image[:,:,::-1].astype(np.single)/255.0
  image = image[None]
  return image, top, left

def process_predictions(pred,image_w,image_h,top,left):
  ## preds are in format [relative x1, relative y1, relative x2, relative y2] [prob] [class] in pred [0, 1 and 2 respectively]
  coords = pred[0][np.nonzero(pred[1]>0)]
  probs = pred[1][np.nonzero(pred[1]>0)]

  # adjust for pad
  coords[:, [0, 2]] = (coords[:, [0, 2]] - left / image_w) / (1 - 2 * left / image_w)
  coords[:, [1, 3]] = (coords[:, [1, 3]] - top / image_h) / (1 - 2 * top / image_h)
  # copy and clip
  new_coords = coords.copy().clip(0,1)
  # re-center
  new_coords [:, 0] = (coords[:,0]+coords[:,2]) / 2
  new_coords [:, 1] = (coords[:, 1] + coords[:, 3]) / 2
  new_coords [:, 2] = (coords[:,2] - coords[:,0])
  new_coords [:, 3] = (coords[:, 3] - coords[:, 1])
  count_preds = new_coords.shape[0]

  # re-pack to class coords, probs
  return np.concatenate([np.zeros(count_preds).reshape(count_preds,1),new_coords,probs.reshape(count_preds,1)],axis=1)



ort_sess = onnxruntime.InferenceSession('./exp13/weights/tf2onnx_best.onnx')

mismatch = 0

for i,f in enumerate(os.listdir('./data/images/train')+os.listdir('./data/images/valid')):
  if os.path.isfile('./data/images/train/%s' % f):
    path = './data/images/train/%s' % f
  elif os.path.isfile('./data/images/valid/%s' % f):
    path = './data/images/valid/%s' % f
  else:
    print("not found in nether thain nor valid %s"% f )
    continue
  image, top, left = prepare_image(path)
  outputs = ort_sess.run(None, {'input_1': image})
  lines  = process_predictions(outputs,image.shape[1],image.shape[2],top, left)

  if os.path.isfile('./yolov5/runs/detect/exp10/labels/%s.txt'%(f.split('.')[0])):
    predictions_file = open('./yolov5/runs/detect/exp10/labels/%s.txt'%(f.split('.')[0]))
    predictions_tr = sorted([[float(i) for i in str.strip(s).split(' ')] for s in predictions_file.readlines()],key=lambda x:x[1])
    predictions_file.close()
    pr_tr_np = np.array(predictions_tr)
    pr_tr_np_s = pr_tr_np[pr_tr_np[:, 1].argsort()]
  else:
    predictions_tr=[]
    pr_tr_np = np.array([])
    pr_tr_np_s = np.array([])

  if len(lines)>0:
    pr_own_np = np.array(lines)[:,:6]
    pr_own_np_s = pr_own_np[pr_own_np[:,1].argsort()]
  else:
    pr_own_np = np.array([])
    pr_own_np_s = np.array([])

  cmp = (pr_tr_np_s.shape == pr_own_np_s.shape)
  if cmp and pr_tr_np_s.shape[0] > 0:
    cmp = np.allclose(pr_tr_np_s,pr_own_np_s,atol=0.01,rtol=0.0)

  if not cmp:
    mismatch += 1
    print(path, " different predictions ")
    print("onnx detect.py", pr_tr_np_s.shape)
    print('\n'.join([str(x) for x in predictions_tr]))
    print("onnx own", pr_own_np_s.shape)
    print('\n'.join([str(x) for x in sorted([[float(i) for i in s] for s in lines],key=lambda x:x[1])]))
    print()

  if (i+1)%15 == 0:
    print("processed:" + str(i) + " mismatch:" + str(mismatch))
#processed:1529 mismatch:15

