from django.db import models


class Tag(models.Model):
    content = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.content


  

class Asset(models.Model):
    symbol = models.CharField(max_length=15)
    platformsymbol = models.CharField(max_length=15, null=True)
    timeframe = models.CharField(max_length=3, null=True)
    identifier = models.CharField(max_length=18, null=True)
    description = models.CharField(max_length=250, null=True)
    name = models.CharField(max_length=50, null=True, blank=True)
    cg_name = models.CharField(max_length=50, null=True, blank=True)
    last_prediction = models.IntegerField(default=0)
    current_prediction = models.IntegerField(default=0)
    ups = models.IntegerField(default=0)
    downs = models.IntegerField(default=0)
    predictions_total = models.IntegerField(default=0)
    predictions_correct = models.IntegerField(default=0)
    accuracy = models.FloatField(default=0)
    sentiment = models.FloatField(default=0)   
    isactive = models.BooleanField(default=True)
    prediction_term = models.DateTimeField(null=True) 
    last_updated = models.DateTimeField(auto_now=True) 
    last_close = models.FloatField(default=0)
    candles = models.TextField(null=True, blank=True) 
    tags = models.ManyToManyField(Tag,null=True, blank=True) 

    def __str__(self):
        return f'{self.identifier}'




class Algorithm(models.Model):
    identifier = models.CharField(max_length=100)
    name = models.CharField(max_length=100, null=True)
    asset = models.ForeignKey(Asset, related_name='models', on_delete=models.CASCADE, null=True)
    criterion = models.CharField(max_length=100, default='entropy', null=True)
    max_depth = models.IntegerField(default=50 , null=True)
    n_estimators = models.IntegerField(default=200, null=True)
    random_state = models.IntegerField(default=42, null=True)
    learning_rate = models.FloatField(default=0, null=True)
    splitter = models.CharField(max_length=15, default='best', null=True)
    isactive = models.BooleanField(default=True)
    accuracy = models.FloatField(default=0)
    prediction = models.IntegerField(default=0)
    predictions_total = models.IntegerField(default=0)
    predictions_correct = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True) 
    last_tuning = models.DateTimeField(null=True) 

    def __str__(self):
        return f'{self.asset} - {self.name}'

class Strategy(models.Model):
    window_s = models.IntegerField(default=9)
    window_m = models.IntegerField(default=14)
    window_l = models.IntegerField(default=50)
    lag = models.IntegerField(default=7)
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, null=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.asset} : {self.window_s}-{self.window_m}-{self.window_l}'

    class Meta:
        verbose_name_plural = "Strategies"

class HitratioHistory(models.Model):
    hitratio = models.IntegerField(null=True)
    asset = models.ForeignKey(Asset, related_name='hitratio', on_delete=models.CASCADE, null=True)


