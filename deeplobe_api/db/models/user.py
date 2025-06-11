from django.db import models

from ..mixins import TimeAuditModel

from django.contrib.auth.models import PermissionsMixin

from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager

from .stripe import StripeSubcription


class UserManager(BaseUserManager):
    use_in_migrations = True

    def _create_user(self, email, password, **extra_fields):
        """
        Creates and saves a User with the given email and password.
        """
        if not email:
            raise ValueError("email must give")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_superuser", False)
        extra_fields.setdefault("is_staff", False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password, **extra_fields):
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_staff", True)

        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(email, password, **extra_fields)


NEW_CHOICES = [
    ("OCR", "ocr"),
    ("IMAGE_CLASSIFICATION", "image_classification"),
    ("IMAGE_SIMILARITY", "image_similarity"),
    ("OBJECT_DETECTION", "object_detection"),
    ("IMAGE_SEGMENTATION", "image_segmentation"),
    ("OTHERS", "others"),
]


class User(AbstractBaseUser, PermissionsMixin, TimeAuditModel):
    username = models.CharField(max_length=255, blank=True)

    email = models.EmailField(unique=True)

    first_name = models.CharField(max_length=255)

    last_name = models.CharField(max_length=255)

    is_staff = models.BooleanField(default=False)

    is_active = models.BooleanField(default=True)

    company = models.CharField(max_length=255, blank=True, null=True)

    job_title = models.CharField(max_length=255, blank=True, null=True)

    terms_conditions = models.BooleanField(default=False)

    token = models.CharField(max_length=256, null=True, blank=True)

    token_status = models.BooleanField(default=False)

    objects = UserManager()

    role = models.CharField(max_length=255, null=True, blank=True)

    industry = models.CharField(max_length=255, null=True, blank=True)

    profile_pic = models.FileField(upload_to="media_file", null=True, blank=True)

    contact_number = models.CharField(max_length=255, blank=True, null=True)

    last_login = models.DateTimeField(null=True, blank=True)

    is_new = models.BooleanField(default=True)

    is_admin = models.BooleanField(default=False)

    model_permissions = models.JSONField(default=list)

    interestedIn = models.JSONField(default=list, null=True, blank=True)

    extended_date = models.DateTimeField(null=True, blank=True)

    current_workspace = models.IntegerField(blank=True, null=True)

    address = models.TextField(max_length=255, null=True, blank=True)

    postal_code = models.IntegerField(blank=True, null=True)

    company_size = models.CharField(max_length=255, blank=True, null=True)

    state = models.CharField(max_length=255, null=True, blank=True)

    country = models.CharField(max_length=255, null=True, blank=True)

    GSTIN = models.TextField(null=True, blank=True)

    bussiness_email = models.CharField(max_length=255, null=True, blank=True)

    help = models.BooleanField(default=True)

    notes = models.TextField(null=True, blank=True)

    stripe_customer_id = models.CharField(max_length=255, null=True, blank=True)
    
    stripe_subcription = models.ForeignKey(
        StripeSubcription, on_delete=models.CASCADE, null=True, blank=True
    )
    
    stripe_currency = models.CharField(max_length=255, null=True, blank=True)
    


    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username", "password"]

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"
        db_table = "users"
