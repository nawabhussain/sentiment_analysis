from pandas import DataFrame
i = 0

def getFiveClassLabels(score):
    rating = int(score)
    if rating<3:
        return "Highly Negative"
    elif 2 < rating < 5:
        return "Negative"
    elif 4 < rating < 7:
        return "Neutral"
    elif 6 < rating < 9:
        return "Positive"
    elif 8 < rating < 11:
        return "Highly Positive"

def getTwoClassLabels(score):
    rating = int(score)
    if rating < 5:
        return "Negative"
    elif 4 < rating < 11:
        return "Positive"

def getThreeClassLabels(score):
    rating = int(score)
    if rating < 5:
        return "Negative"
    elif 4 < rating < 7:
        return "Neutral"
    elif 6 < rating < 11:
        return "Positive"

list = []
with open("SAR14.txt") as fp:
    for line in fp:
        # print(line)
        indexStart = line.index("\"")
        indexEnd = line.rfind("\" ,")
        comment = line[(indexStart+1):indexEnd]
        rating = line[(indexEnd+3):len(line)].rstrip()
        label = getFiveClassLabels(rating)
        # label = getThreeClassLabels(rating)
        # label = getTwoClassLabels(rating)
        i = i +1
        list.append({'Comment': comment,'Rating':rating,'Label':label})
        # if(i == 2):
        #     break;
    # print(list)
    df = DataFrame(list)
    # print(df)
    df.to_csv("Dataset.csv", sep=',', encoding='utf-8')
    print("Done")
