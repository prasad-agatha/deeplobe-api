from .base import BaseModelSerializer

from rest_framework import serializers

from deeplobe_api.db.models import AIModel, AIModelPrediction, APILogger


def get_job_status(job_id):
    from django_rq import get_connection
    from rq.job import Job

    con = get_connection("default")
    job = Job.fetch(job_id, connection=con)
    status = job.get_status()
    return status


class CustomModelCreationSerializer(BaseModelSerializer):
    stats_count = serializers.SerializerMethodField(read_only=True)
    email = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = AIModel
        fields = "__all__"

    def get_stats_count(self, object):
        return APILogger.objects.filter(model_id=object).count()

    def get_email(self, object):
        return object.user.email


class CustomModelUpdationSerializer(BaseModelSerializer):
    class Meta:
        model = AIModel
        fields = ("name", "description", "annotation_file")


class AIModelSerializer(BaseModelSerializer):
    # job = serializers.SerializerMethodField(read_only=True)
    stats_count = serializers.SerializerMethodField(read_only=True)
    email = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = AIModel
        fields = "__all__"

    def get_job(self, object):
        job_id = object.job.get("id")
        if job_id is not None:
            job = object.job
            try:
                job_status = get_job_status(job_id)
                job["status"] = job_status
            except:
                job["status"] = ""
            return job
        else:
            return object.job

    def get_stats_count(self, object):
        return APILogger.objects.filter(model_id=object).count()

    def get_email(self, object):
        return object.user.email


class AIModelPredictionSerializer(BaseModelSerializer):
    class Meta:
        model = AIModelPrediction
        fields = "__all__"


class CustomAllModelSerializer(BaseModelSerializer):
    class Meta:
        model = AIModel
        fields = ("id", "name", "model_type", "status")
