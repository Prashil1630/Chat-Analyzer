
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob  # For sentiment analysis

extractor = URLExtract()

def fetch_stat(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))
    return num_messages, len(words), num_media_messages, len(links)

def most_busy_user(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts()/df.shape[0])*100, 2).reset_index().rename(columns={'user': 'name', 'count': 'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().splitlines()
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    def remove_stop_words(message):
        return " ".join(word for word in message.lower().split() if word not in stop_words)
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().splitlines()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    return_df = pd.DataFrame(Counter(words).most_common(20))
    return return_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = [c for message in df['message'] for c in message if emoji.is_emoji(c)]
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'].astype(str) + "-" + timeline['year'].astype(str)
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def user_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    activity_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return activity_heatmap

def sentiment_analysis(daily_timeline, df):
    sentiment_data = daily_timeline.copy()
    sentiment_data['positive'] = 0
    sentiment_data['negative'] = 0
    sentiment_data['neutral'] = 0  # Add neutral column

    if 'only_date' not in sentiment_data.columns:
        raise KeyError("Column 'only_date' not found in daily_timeline DataFrame")
    if 'message' not in df.columns:
        raise KeyError("Column 'message' not found in df DataFrame")

    for i, row in sentiment_data.iterrows():
        date = row['only_date']
        messages = df[df['only_date'] == date]['message']
        sentiment_scores = [TextBlob(message).sentiment.polarity for message in messages]
        sentiment_data.at[i, 'positive'] = sum(score > 0 for score in sentiment_scores)
        sentiment_data.at[i, 'negative'] = sum(score < 0 for score in sentiment_scores)
        sentiment_data.at[i, 'neutral'] = sum(score == 0 for score in sentiment_scores)

    return sentiment_data
