# from django.db import models
# from django.db.models.fields import json
# from ..mixins import TimeAuditModel
# from django.db.models import JSONField
# from django.conf import settings


# class ChargeBeeModel(TimeAuditModel):

#     email = models.JSONField(default=list)
#     customer_id = models.CharField(max_length=255, blank=False)

#     class Meta:
#         verbose_name = "ChargeBee"
#         verbose_name_plural = " ChargeBees"
#         db_table = "chargebee"

#     def __str__(self):
#         return f"{self.email}"
