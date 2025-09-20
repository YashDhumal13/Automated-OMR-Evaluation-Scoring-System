# template_maker.py
# Usage: python template_maker.py path/to/blank_sheet.jpg templates/template_v1.json
import cv2, json, sys, os

if len(sys.argv) < 3:
    print("Usage: python template_maker.py blank.jpg out.json"); sys.exit(1)

img_path = sys.argv[1]
out_path = sys.argv[2]
img = cv2.imread(img_path)
h,w = img.shape[:2]
clone = img.copy()
points = []

def click(event,x,y,flags,param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x/w, y/h))
        cv2.circle(img, (x,y), 8, (0,255,0), -1)
        cv2.imshow("img", img)

cv2.namedWindow("img")
cv2.setMouseCallback("img", click)
print("Click bubble centers in order. Press 'q' when done.")
while True:
    cv2.imshow("img", img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
cv2.destroyAllWindows()

# Save normalized points
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path,"w") as f:
    json.dump({"w": w, "h": h, "points": points}, f, indent=2)
print("Saved", out_path, "with", len(points), "points")
